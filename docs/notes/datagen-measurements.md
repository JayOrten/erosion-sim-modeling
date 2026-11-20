# Datagen measurements (implementation plan, step 8)

Measured 2026-11-18 on the laptop (one process), with `datagen/src/erosion_datagen/measure.py`.

## Feature size per noise combination

Dominant wavelength: where wavenumber × power peaks in the radial spectrum of the start terrain. 512²
terrains at period 48, octaves 5, persistence 0.5, lacunarity 2.1; median of 12 seeds.

| noise / fractal / warp        | cells | × period | p90 / p10 |
| ----------------------------- | ----- | -------- | --------- |
| perlin / fbm / no warp        | 73    | 1.52     | 1.58      |
| perlin / fbm / warp           | 73    | 1.52     | 1.60      |
| perlin / ridged / no warp     | 47    | 0.97     | 1.38      |
| perlin / ridged / warp        | 64    | 1.33     | 2.16      |
| opensimplex2 / fbm / no warp  | 47    | 0.97     | 1.20      |
| opensimplex2 / fbm / warp     | 54    | 1.13     | 1.36      |
| opensimplex2 / ridged / no warp | 20  | 0.41     | 1.00      |
| opensimplex2 / ridged / warp  | 34    | 0.71     | 1.69      |
| value_cubic / fbm / no warp   | 128   | 2.67     | 2.42      |
| value_cubic / fbm / warp      | 128   | 2.67     | 2.42      |
| value_cubic / ridged / no warp | 107  | 2.22     | 2.33      |
| value_cubic / ridged / warp   | 128   | 2.67     | 3.33      |

At the same `period`, feature size spans 0.4× to 2.7× the period: a factor of about 6.5 between
combinations. The spectrum bins are coarse at long wavelengths (512 / 4 = 128), so the value-noise rows
are approximate.

## How far the fixed edges reach

A region simulated alone (edges fixed) against the same region as the middle of a run twice as wide.
Ratio: RMS difference of the final height change over RMS change, by distance from the region's edge.
50,000 yr, `K_ref` 2.2e-4 unless noted, `m` 0.45.

256² alone against the middle of 512²:

| case                                 | d = 8 | 16   | 32   | 64   | 96   |
| ------------------------------------ | ----- | ---- | ---- | ---- | ---- |
| period 32, D/K 1                     | 0.20  | 0.18 | 0.17 | 0.12 | 0.08 |
| period 64, D/K 1                     | 0.29  | 0.26 | 0.23 | 0.32 | 0.35 |
| period 48, K_ref 5e-4, D/K 0.3       | 0.12  | 0.11 | 0.12 | 0.15 | 0.13 |
| period 48, K_ref 1e-4, D/K 3         | 0.32  | 0.29 | 0.28 | 0.23 | 0.11 |

512² alone against the middle of 1024² (mean over 16-cell distance bands):

| case      | 0–16 | 32–48 | 64–80 | 96–112 | 128–144 | 160–176 | 192–208 | 224–240 |
| --------- | ---- | ----- | ----- | ------ | ------- | ------- | ------- | ------- |
| period 32 | 0.30 | 0.18  | 0.17  | 0.11   | 0.13    | 0.11    | 0.08    | 0.03    |
| period 64 | 0.37 | 0.18  | 0.18  | 0.15   | 0.13    | 0.14    | 0.10    | 0.12    |

Not sensitivity to small changes: a 1 mm bump at one cell of a 256² run changes nothing more than 100
cells away after 50,000 yr. The edges act as outlets, so basins near them drain differently, and the
drainage change reaches across the whole region. A difference of 10–15% remains 200 cells from the
edge.

## Simulator speed

One process on the laptop, M cell-steps per second: 4.8 at 256², 3.1 at 512², 2.5 at 1024², 1.6 at
2048². Larger grids are slower per cell. The plan's 27 M was the total over 16 processes on this
machine, so 16 processes gave about 5.6 times one: memory bandwidth limits the scaling.

CPU time per group at these rates (one process per run):

| group | runs | CPU-hours |
| ----- | ---- | --------- |
| `main_256` | 4,800 | 9 |
| `main_512` | 3,600 | 42 |
| `main_1024` | 1,500 | 88 |
| `main_2048` | 400 | 149 |
| `ood_256` | 200 | 0 |
| `ood_512` | 200 | 2 |
| `long_256` | 200 | 1 |
| `long_512` | 200 | 7 |
| **total** | | **300** |

On a 32-core node with perfect scaling that is about 9 hours; with the laptop's scaling
(16 processes ≈ 5.6 × one) it would be much longer. A short timed run on a cluster node gives the real
figure before the full batches.
