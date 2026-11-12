"""A browser viewer for the infinite world: pan and zoom over eroded terrain, computed as you go.

Run it, then open the printed address:

    uv run python -m erosion_sim.world.viewer              # loads artifacts/checkpoints/diffusion-world_*.pt
    uv run python -m erosion_sim.world.viewer --steps 32   # full sampler steps: slower, as in the notebook

The page (``viewer.html``) is a Leaflet map with two layers of 256-pixel tiles:

- The starting terrain, everywhere and at any zoom. It is plain noise, so it renders at once.
- On top, the eroded terrain (or ``Δz``), computed by the world as tiles come into view: about 1 to 3 s
  per new tile on a GPU. Zoomed in (1 and 2 m per pixel) it comes from the fine level, zoomed out
  (4 and 8 m per pixel) from the coarse level. Further out, only the starting terrain shows.

The page keeps its own queue: at most two eroded tiles are computed at a time, nearest the view center
first, and tiles that leave the view are dropped. Colors and shading use fixed ranges, so neighboring
tiles match. The models load once; the server keeps the last few worlds (one per seed, parameters, and
``dT``) with their caches, so revisiting a place is instant. By default the viewer samples with 16 steps
in 16-bit arithmetic, which is about 2.5 times faster than the notebook's 32 steps in 32-bit.
"""

import argparse
import io
import json
import socket
import threading
import time
from collections import OrderedDict, deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import matplotlib.pyplot as plt
import numpy as np
from erosion_terrain import Terrain
from matplotlib import colormaps

from ..paths import checkpoint_path
from .data import block_mean, coarse_heights
from .world import World, load_levels

TILE = 256
# Leaflet zoom z shows 2 ** -z m per pixel. Eroded tiles exist from MIN_MODEL_ZOOM to 0; the starting
# terrain from MIN_ZOOM. The terrain has no features above its 32 m period, so zooming out further shows
# only uniform texture.
MIN_ZOOM, MIN_MODEL_ZOOM, MAX_NATIVE_ZOOM = -4, -3, 0
# Fixed color ranges (m), so every tile uses the same colors.
HEIGHT_RANGE = (-70.0, 70.0)
CHANGE_RANGE = 40.0
LAYERS = ("eroded", "start", "change")
DEFAULTS = {"seed": 1, "dT": 30_000.0, "K_ref": 2.5e-4, "D_over_K": 1.0, "m": 0.45, "persistence": 0.5}
PAGE = Path(__file__).with_name("viewer.html")


def shade(z: np.ndarray, pixel: float) -> np.ndarray:
    """Light intensity in [0, 1] of heights ``z`` (padded by one cell on each side) with cells ``pixel`` m wide.

    Fixed light from the upper left, no per-image contrast stretch, so tiles shade the same at their
    shared edges. Returns the interior, two cells smaller in each axis.
    """
    gy, gx = np.gradient(z, pixel)
    gy, gx = gy[1:-1, 1:-1], gx[1:-1, 1:-1]
    light = np.array([-1.0, -1.0, 1.0]) / np.sqrt(3)  # from the upper left, 35° up
    normal = np.stack([-gx, -gy, np.ones_like(gx)])
    normal /= np.linalg.norm(normal, axis=0)
    return np.clip(np.tensordot(light, normal, axes=1), 0, 1)


def color(values: np.ndarray, layer: str) -> np.ndarray:
    """RGB in [0, 1] for heights or changes (m), with the fixed ranges."""
    if layer == "change":
        return colormaps["RdBu_r"](np.clip(values / (2 * CHANGE_RANGE) + 0.5, 0, 1))[..., :3]
    low, high = HEIGHT_RANGE
    return colormaps["terrain"](np.clip((values - low) / (high - low), 0, 1))[..., :3]


def png(heights: np.ndarray, values: np.ndarray, layer: str, pixel: float) -> bytes:
    """A tile: ``values`` (the interior of ``heights``) colored, shaded by ``heights``."""
    rgb = color(values, layer) * (0.35 + 0.65 * shade(heights, pixel))[..., None]
    buffer = io.BytesIO()
    plt.imsave(buffer, np.clip(rgb, 0, 1), format="png")
    return buffer.getvalue()


def client_gone(connection: socket.socket) -> bool:
    """Whether the browser closed this request's connection (it does when a tile is cancelled)."""
    try:
        connection.setblocking(False)
        return connection.recv(1, socket.MSG_PEEK) == b""
    except BlockingIOError:
        return False
    except OSError:
        return True
    finally:
        connection.setblocking(True)


class Tiles:
    """Builds worlds on demand and renders their tiles.

    ``levels`` are the two levels' ``(model, scaling, config)``, from ``load_levels``. ``keep`` worlds
    (one per seed, parameters, and ``dT``) stay cached.
    """

    def __init__(self, levels: list[tuple], steps: int, bf16: bool, batch_size: int, keep: int = 4):
        self.levels = list(levels)
        self.block = self.levels[1][2].block
        self.sampler = {"steps": steps, "bf16": bf16, "batch_size": batch_size}
        self.worlds: OrderedDict = OrderedDict()
        self.keep = keep
        self.lock = threading.Lock()  # the models and the caches are used by one tile at a time
        self.status = {"computing": None, "done": 0, "skipped": 0, "recent": deque(maxlen=20)}

    def world(self, settings: dict) -> World:
        key = tuple(sorted(settings.items()))
        if key in self.worlds:
            self.worlds.move_to_end(key)
            return self.worlds[key]
        s = settings
        world = World(int(s["seed"]), s["dT"], *self.levels, K_ref=s["K_ref"], D=s["K_ref"] * s["D_over_K"],
                      m=s["m"], persistence=s["persistence"], **self.sampler)
        self.worlds[key] = world
        if len(self.worlds) > self.keep:
            self.worlds.popitem(last=False)
        return world

    def grid(self, world: World, pixel: int, y0: int, x0: int, n: int) -> tuple[np.ndarray, np.ndarray]:
        """``(start, change)`` (m) on ``n`` × ``n`` cells ``pixel`` m wide, starting at meter (``y0``, ``x0``).

        Below ``block`` m per pixel this reads the fine level, otherwise the coarse level (``block`` m cells),
        averaging its cells down to the pixel size.
        """
        unit = 1 if pixel < self.block else self.block  # meters per cell of the level used
        factor, cy, cx = pixel // unit, y0 // unit, x0 // unit
        size = n * factor
        if unit == 1:
            start, change = world.terrain.heights(cy, cx, size, size), world.change(cy, cx, size, size)
        else:
            start = coarse_heights(world.terrain, cy, cx, size, size, block=self.block)
            change = world.coarse_change(cy, cx, size, size)
        if factor > 1:
            start, change = block_mean(start, factor), block_mean(change, factor)
        return start, change

    def start_tile(self, settings: dict, zoom: int, tx: int, ty: int) -> bytes:
        """A starting-terrain tile. Needs no model, so it is rendered right away, at any zoom."""
        pixel = 2.0 ** -zoom
        terrain = Terrain(int(settings["seed"]), persistence=settings["persistence"])
        y0, x0 = int((ty * TILE - 1) * pixel), int((tx * TILE - 1) * pixel)
        # One noise sample per pixel, at the center of the pixel's cells: a preview, cheap at any zoom.
        # Not at a corner: Perlin noise is 0 on its lattice, and every octave's lattice divides 2^k meters.
        step = int(pixel)
        center = (step - 1) / 2
        heights = terrain.heights(y0 + center, x0 + center, TILE + 2, TILE + 2, step=step)
        return png(heights, heights[1:-1, 1:-1], "start", pixel)

    def model_tile(self, settings: dict, layer: str, zoom: int, tx: int, ty: int, connection=None) -> bytes | None:
        """An eroded or ``Δz`` tile; ``None`` if the browser cancelled it while it waited."""
        pixel = 2 ** -zoom
        # One extra pixel on every side, so the shading at the tile edge sees its neighbors.
        y0, x0 = (ty * TILE - 1) * pixel, (tx * TILE - 1) * pixel
        with self.lock:
            if connection is not None and client_gone(connection):
                self.status["skipped"] += 1
                return None
            name = f"{layer} {pixel} m/px tile ({tx}, {ty})"
            self.status["computing"] = name
            start = time.perf_counter()
            world = self.world(settings)
            start_heights, change = self.grid(world, pixel, y0, x0, TILE + 2)
            heights = start_heights + change
            values = (change if layer == "change" else heights)[1:-1, 1:-1]
            seconds = time.perf_counter() - start
            self.status.update(computing=None, done=self.status["done"] + 1)
            self.status["recent"].append(seconds)
        print(f"{name}: {seconds:.1f} s", flush=True)
        return png(heights, values, layer, pixel)

    def report(self) -> dict:
        recent = list(self.status["recent"])
        return {"computing": self.status["computing"], "done": self.status["done"], "skipped": self.status["skipped"],
                "seconds_per_tile": float(np.mean(recent)) if recent else None}


def parse_settings(query: dict) -> dict:
    settings = dict(DEFAULTS)
    for name, default in DEFAULTS.items():
        if name in query:
            settings[name] = type(default)(float(query[name][0]))
    return settings


def make_handler(tiles: Tiles):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_GET(self):
            url = urlparse(self.path)
            parts = url.path.strip("/").split("/")
            try:
                if url.path in ("/", "/index.html"):
                    self.send(PAGE.read_bytes(), "text/html; charset=utf-8")
                elif url.path == "/config":
                    config = {"defaults": DEFAULTS, "layers": LAYERS, "min_zoom": MIN_ZOOM,
                              "min_model_zoom": MIN_MODEL_ZOOM, "max_native_zoom": MAX_NATIVE_ZOOM,
                              "block": tiles.block, "steps": tiles.sampler["steps"]}
                    self.send(json.dumps(config).encode(), "application/json")
                elif url.path == "/status":
                    self.send(json.dumps(tiles.report()).encode(), "application/json")
                elif parts[0] == "tile" and len(parts) == 5 and parts[1] in LAYERS:
                    layer, zoom, tx, ty = parts[1], int(parts[2]), int(parts[3]), int(parts[4].removesuffix(".png"))
                    settings = parse_settings(parse_qs(url.query))
                    if layer == "start":
                        if not MIN_ZOOM <= zoom <= MAX_NATIVE_ZOOM:
                            raise ValueError(f"zoom must be from {MIN_ZOOM} to {MAX_NATIVE_ZOOM}")
                        body = tiles.start_tile(settings, zoom, tx, ty)
                    else:
                        if not MIN_MODEL_ZOOM <= zoom <= MAX_NATIVE_ZOOM:
                            raise ValueError(f"zoom must be from {MIN_MODEL_ZOOM} to {MAX_NATIVE_ZOOM}")
                        body = tiles.model_tile(settings, layer, zoom, tx, ty, self.connection)
                        if body is None:
                            self.close_connection = True
                            return
                    self.send(body, "image/png", cache=True)
                else:
                    self.send(b"not found", "text/plain", status=404)
            except (BrokenPipeError, ConnectionResetError):
                self.close_connection = True  # the browser stopped waiting for this tile
            except Exception as error:  # noqa: BLE001 - report any failure to the page and the terminal
                print(f"error on {self.path}: {error!r}", flush=True)
                self.send(str(error).encode(), "text/plain", status=400)

        def send(self, body: bytes, kind: str, status: int = 200, cache: bool = False):
            self.send_response(status)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "max-age=86400" if cache else "no-store")
            self.end_headers()
            self.wfile.write(body)

        def handle(self):
            try:
                super().handle()
            except (BrokenPipeError, ConnectionResetError):
                pass  # a cancelled tile's connection, closed by the browser

        def log_message(self, format, *args):
            pass

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--coarse", type=Path, default=checkpoint_path("world_coarse", "diffusion"))
    parser.add_argument("--fine", type=Path, default=checkpoint_path("world_fine", "diffusion"))
    parser.add_argument("--device", default=None, help="cuda or cpu; default cuda if available")
    parser.add_argument("--steps", type=int, default=16, help="sampler steps per level (the notebook uses 32)")
    parser.add_argument("--fp32", action="store_true", help="run the models in 32-bit instead of 16-bit")
    parser.add_argument("--batch-size", type=int, default=64, help="windows per model call")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    tiles = Tiles(load_levels(args.coarse, args.fine, args.device), args.steps, not args.fp32, args.batch_size)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(tiles))
    server.daemon_threads = True
    print(f"world viewer at http://{args.host}:{args.port}  (Ctrl-C to stop)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
