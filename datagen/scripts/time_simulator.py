"""Step 1 of ../docs/unet-baseline-plan.md: time and profile one Landlab erosion run.

Run from datagen/ with: uv run python scripts/time_simulator.py
"""

import time

import numpy as np
from landlab import RasterModelGrid
from landlab.components import FastscapeEroder, FlowAccumulator, LinearDiffuser

from erosion_datagen.terrain import fbm

SHAPE = (128, 128)
HEIGHT = 64.0
DT = 100.0
STEPS = 500


def build(z0, K, D, m, route_pits):
    grid = RasterModelGrid(z0.shape, xy_spacing=1.0)
    grid.add_field("topographic__elevation", z0.ravel().copy(), at="node")
    flow = FlowAccumulator(
        grid,
        flow_director="D8",
        depression_finder="DepressionFinderAndRouter" if route_pits else None,
    )
    incision = FastscapeEroder(grid, K_sp=K, m_sp=m, n_sp=1.0)
    diffusion = LinearDiffuser(grid, linear_diffusivity=D)
    return grid, flow, incision, diffusion


def profile_run(z0, K=1e-3, D=1e-3, m=0.4, U=0.0, route_pits=True, steps=STEPS):
    """Run once and time each part of every step. Return per-part totals in seconds and per-step stats."""
    grid, flow, incision, diffusion = build(z0, K, D, m, route_pits)
    z = grid.at_node["topographic__elevation"]
    core = grid.core_nodes
    parts = {"uplift": 0.0, "flow": 0.0, "incision": 0.0, "diffusion": 0.0}
    change, pits = [], []

    start = time.perf_counter()
    for _ in range(steps):
        t0 = time.perf_counter()
        z[core] += U * DT
        t1 = time.perf_counter()
        flow.run_one_step()
        t2 = time.perf_counter()
        incision.run_one_step(DT)
        t3 = time.perf_counter()
        diffusion.run_one_step(DT)
        t4 = time.perf_counter()
        parts["uplift"] += t1 - t0
        parts["flow"] += t2 - t1
        parts["incision"] += t3 - t2
        parts["diffusion"] += t4 - t3
        change.append(np.abs(z - z0.ravel()).mean())
        if route_pits:
            pits.append(flow.depression_finder.number_of_lakes)
    total = time.perf_counter() - start
    parts["other"] = total - sum(parts.values())
    return total, parts, np.array(change), np.array(pits)


def report(label, total, parts, steps=STEPS):
    print(f"{label}: {total:.2f} s total, {total / steps * 1000:.2f} ms/step")
    for name, seconds in parts.items():
        print(f"    {name:<10} {seconds / steps * 1000:7.3f} ms/step  ({seconds / total:5.1%})")


def main():
    z0 = HEIGHT * fbm(SHAPE, seed=0)

    # 1. The default settings, with and without pit routing.
    total, parts, change, pits = profile_run(z0)
    report("pit routing on", total, parts)
    print("    lakes routed at steps 1, 10, 100, 500:", pits[[0, 9, 99, 499]])
    print("    mean |change| (m) at steps 100..500:", np.round(change[99::100], 2))
    total_off, parts_off, _, _ = profile_run(z0, route_pits=False)
    report("pit routing off", total_off, parts_off)

    # 2. Diffusion substeps. LinearDiffuser splits dt into substeps when D * dt exceeds its stability limit.
    print("\ndiffusion cost against D (100 steps each):")
    for D in [1e-4, 1e-3, 1e-2, 1e-1]:
        _, parts_d, _, _ = profile_run(z0, D=D, steps=100)
        print(f"    D={D:g}: diffusion {parts_d['diffusion'] / 100 * 1000:7.3f} ms/step")

    # 3. Does terrain style or uplift change the cost? Uplift keeps making relief, so pits may persist.
    print("\nother settings (100 steps each):")
    for label, z_start, kwargs in [
        ("rough terrain (period 32, persistence 0.7)", HEIGHT * fbm(SHAPE, seed=1, period=32, persistence=0.7), {}),
        ("uplift 1e-3 m/yr", z0, dict(U=1e-3)),
        ("K=1e-2", z0, dict(K=1e-2)),
    ]:
        total_s, _, _, _ = profile_run(z_start, steps=100, **kwargs)
        print(f"    {label}: {total_s / 100 * 1000:.2f} ms/step")

    print(f"\none 500-step run with pit routing: {total:.1f} s")
    for runs in [2_000, 5_000, 10_000]:
        print(f"    {runs:>6} runs on 32 cores: about {runs * total / 32 / 60:.0f} min (assumes linear scaling)")


if __name__ == "__main__":
    main()
