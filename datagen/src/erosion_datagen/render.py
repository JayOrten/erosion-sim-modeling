"""3D terrain rendering with PyVista."""

import numpy as np
import pyvista as pv
from matplotlib import colormaps
from matplotlib.colors import LinearSegmentedColormap, Normalize

# matplotlib's "terrain" colormap without its blue low end, so low ground reads as land, not water.
LAND = LinearSegmentedColormap.from_list("land", colormaps["terrain"](np.linspace(0.25, 1.0, 256)))
RIVER = np.array([0.20, 0.42, 0.70])


def terrain_mesh(
    z: np.ndarray,
    spacing: float = 1.0,
    vert_exag: float = 1.0,
    base_depth: float = 8.0,
    bottom: float | None = None,
) -> pv.PolyData:
    """Turn a 2D heightmap into a closed block: the terrain surface on top, flat sides, and a flat bottom.

    The bottom sits ``base_depth`` below the lowest point, or at height ``bottom`` (before ``vert_exag``) if given.
    """
    rows, cols = z.shape
    y, x = np.mgrid[0:rows, 0:cols] * spacing
    points = np.column_stack([x.ravel(), y.ravel(), z.ravel() * vert_exag])

    # One quad per grid cell, with corners listed by node index (row * cols + col).
    corner = (np.arange(rows - 1)[:, None] * cols + np.arange(cols - 1)[None, :]).ravel()
    quads = np.column_stack([np.full_like(corner, 4), corner, corner + 1, corner + cols + 1, corner + cols])
    surface = pv.PolyData(points, faces=quads.ravel())

    floor_z = (z.min() if bottom is None else bottom) * vert_exag - base_depth
    floor = pv.Plane(
        center=((cols - 1) * spacing / 2, (rows - 1) * spacing / 2, floor_z),
        i_size=cols * spacing * 2, j_size=rows * spacing * 2,
    )
    return surface.extrude_trim((0, 0, -1), floor)


def terrain_colors(
    z: np.ndarray,
    area: np.ndarray | None = None,
    river_threshold: float = 3.0,
    z_range: tuple[float, float] | None = None,
) -> np.ndarray:
    """RGB color for each node: elevation through the LAND colormap, blended toward blue along rivers.

    Colors span ``z_range`` (low, high), or the heightmap's own range if not given. ``area`` is drainage
    area in m². Nodes with log10(area) above ``river_threshold`` fade toward blue.
    """
    low, high = z_range if z_range is not None else (z.min(), z.max())
    rgb = LAND(Normalize(low, high, clip=True)(z))[..., :3]
    if area is not None:
        strength = np.clip(np.log10(np.maximum(area, 1.0)) - river_threshold, 0.0, 1.0)[..., None]
        rgb = (1 - strength) * rgb + strength * RIVER
    return rgb.reshape(-1, 3)


def _colored_block(z, area, spacing, vert_exag, river_threshold, z_range, bottom=None) -> pv.PolyData:
    block = terrain_mesh(z, spacing, vert_exag, bottom=bottom)
    top = terrain_colors(z, area, river_threshold, z_range)
    # extrude_trim keeps the original surface points first, so the top surface gets the terrain
    # colors and the added side and bottom points get a plain soil color.
    colors = np.tile([0.45, 0.36, 0.28], (block.n_points, 1))
    colors[: top.shape[0]] = top
    block["rgb"] = colors
    return block


def _plotter(block, shape, spacing, window_size, azimuth, elevation, zoom, interactive=False) -> tuple[pv.Plotter, pv.Actor]:
    """A lit plotter showing ``block``, with the camera fitted to it. Return the plotter and the mesh actor."""
    plotter = pv.Plotter(off_screen=not interactive, window_size=window_size, lighting="none")
    actor = plotter.add_mesh(block, scalars="rgb", rgb=True, smooth_shading=True, ambient=0.25, diffuse=0.85, specular=0.0)
    plotter.add_light(pv.Light(position=(-1, 1, 1.2), light_type="scene light", intensity=1.0))
    plotter.add_light(pv.Light(position=(1, -0.5, 0.6), light_type="scene light", intensity=0.25))
    plotter.set_background("#e9eef3", top="#9fb8d0")
    plotter.enable_ssao(radius=max(shape) * spacing * 0.02)

    plotter.view_isometric()
    plotter.camera.azimuth = azimuth
    plotter.camera.elevation = elevation
    plotter.reset_camera()
    plotter.camera.zoom(zoom)
    return plotter, actor


def render_terrain(
    z: np.ndarray,
    area: np.ndarray | None = None,
    spacing: float = 1.0,
    vert_exag: float = 1.0,
    river_threshold: float = 3.0,
    window_size: tuple[int, int] = (1400, 900),
    azimuth: float = -35.0,
    elevation: float = 25.0,
    zoom: float = 1.15,
    interactive: bool = False,
    z_range: tuple[float, float] | None = None,
) -> np.ndarray | None:
    """Render a heightmap as a lit 3D block.

    By default this renders off-screen and returns the image as an array (show it with ``plt.imshow``).
    With ``interactive=True`` it opens a window you can rotate with the mouse, and returns None.
    """
    block = _colored_block(z, area, spacing, vert_exag, river_threshold, z_range)
    plotter, _ = _plotter(block, z.shape, spacing, window_size, azimuth, elevation, zoom, interactive)
    if interactive:
        plotter.show()
        return None
    image = plotter.screenshot(return_img=True)
    plotter.close()
    return image


def render_frames(
    zs: np.ndarray,
    areas: np.ndarray | None = None,
    spacing: float = 1.0,
    vert_exag: float = 1.0,
    river_threshold: float = 3.0,
    window_size: tuple[int, int] = (1000, 700),
    azimuth: float = -35.0,
    elevation: float = 25.0,
    zoom: float = 1.15,
) -> list[np.ndarray]:
    """Render a sequence of heightmaps ``zs`` (frames, rows, cols) as images, for an animation.

    Every frame shares one camera, one color scale, and one block bottom, so the only thing that changes
    between frames is the terrain.
    """
    z_range = (float(zs.min()), float(zs.max()))
    bottom = z_range[0]
    tallest = int(np.argmax(zs.max(axis=(1, 2))))  # fit the camera to the tallest frame so none is cut off
    block = _colored_block(zs[tallest], None if areas is None else areas[tallest], spacing, vert_exag, river_threshold, z_range, bottom)
    plotter, actor = _plotter(block, zs.shape[1:], spacing, window_size, azimuth, elevation, zoom)
    images = []
    for i, z in enumerate(zs):
        plotter.remove_actor(actor, render=False)
        block = _colored_block(z, None if areas is None else areas[i], spacing, vert_exag, river_threshold, z_range, bottom)
        actor = plotter.add_mesh(block, scalars="rgb", rgb=True, smooth_shading=True, ambient=0.25, diffuse=0.85, specular=0.0, reset_camera=False)
        images.append(plotter.screenshot(return_img=True))
    plotter.close()
    return images
