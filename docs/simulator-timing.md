# Simulator timing (step 1)

Measured with `datagen/scripts/time_simulator.py`. Grid 128×128, cell size 1 m, `dt = 100` yr, terrain
`64 * fbm(seed=0)`, `K = 1e-3`, `D = 1e-3`, `m = 0.4`, `n = 1`, no uplift, one process.

## Where the time goes

Pit routing takes almost all of it.

| | ms/step, pits routed | ms/step, pits not routed |
|---|---|---|
| flow routing (`FlowAccumulator`) | 516.3 | 2.3 |
| incision (`FastscapeEroder`) | 0.8 | 1.7 |
| diffusion (`LinearDiffuser`) | 2.1 | 4.5 |
| uplift and other overhead | 0.1 | 0.1 |
| **total** | **519** | **8.5** |

A 500-step run takes 260 s with pit routing and 4 s without.

## Why pit routing is slow

`DepressionFinderAndRouter` is mostly pure Python. Its cost grows with the number of nodes inside
lakes, not the number of lakes. In the default run, the lakes shrink in number but one grows large:

| step | flow routing (ms) | lakes | nodes in lakes |
|---|---|---|---|
| 1 | 51 | 21 | 2,052 |
| 100 | 64 | 3 | 2,094 |
| 200 | 218 | 1 | 5,024 |
| 300 | 554 | 1 | 8,917 |

By step 300, more than half of the 16,384 nodes are in one "lake". This is the interior worn down
to a near-flat floor at the level of the lowest edge. The terrain is mostly done eroding by then: the
mean height change is 13.3 m at step 100, 19.4 m at step 200, and 21.9 m at step 500.

So the slow runs are also the over-eroded ones. The same pattern shows in other settings, measured
over 100 steps:

| setting | ms/step |
|---|---|
| default | 64 (over the first 100 steps) |
| `K = 1e-2` (flattens fast) | 971 |
| uplift `1e-3` m/yr (keeps relief) | 22 |
| rough terrain (`period = 32`, `persistence = 0.7`) | 75 |

## Diffusion substeps

`LinearDiffuser` splits each step into substeps when `D·dt` exceeds its stability limit (substep
length `0.15 / D` yr at 1 m cells). Its cost per step, over 100 steps:

| `D` | ms/step |
|---|---|
| 1e-4 | 1.7 |
| 1e-3 | 1.1 |
| 1e-2 | 6.3 |
| 1e-1 | 444 |

Up to `D = 1e-2` this is cheap. Above that, cost grows linearly with `D`.

## Full dataset cost

On 32 cores, assuming linear scaling:

| runs | pits routed (current) | pits not routed |
|---|---|---|
| 2,000 | 4.5 h | 4 min |
| 5,000 | 11 h | 11 min |
| 10,000 | 23 h | 22 min |

The pit-routed estimate uses the default run's 260 s. Runs that flatten faster would cost more.

## Alternatives to `DepressionFinderAndRouter`

**`PriorityFloodFlowRouter` (Landlab): rejected.** It needs the `richdem` package. The last `richdem`
release is 0.3.4 from 2019. It has wheels only up to Python 3.7, and it requires `numpy<2`. This
project uses Python 3.13 and numpy 2, so it cannot be installed.

**`LakeMapperBarnes` (Landlab): no faster.** It uses priority-flood, but in pure Python. Over the
first 200 steps, flow routing plus pit handling averaged 137 ms/step, against 91 ms/step for
`DepressionFinderAndRouter` and 1.6 ms/step with no pit handling.

**No pit routing: close to the pit-routed result.** After 200 steps of the default run:

| compared with `DepressionFinderAndRouter` | RMSE (m) | max difference (m) |
|---|---|---|
| no pit routing | 2.43 | 5.19 |
| `LakeMapperBarnes` | 2.08 | 5.71 |

The two pit-routing methods differ from each other about as much as no pit routing differs from
`DepressionFinderAndRouter`. The terrain relief is 64 m. Visually, all three have a connected channel network and the same
overall shape. The differences come from the main channels taking different paths early in the run,
when the starting terrain still has many pits (21 at step 1, 3 by step 100).

`DepressionFinderAndRouter` also produces an artifact. Once a large flat floor forms, it routes water
across the flat in straight parallel lines.

## Threads

By default, Landlab's flow routing, incision, and diffusion each spread over about 30 threads. At
this grid size that makes them slower. Over 200 steps, `K = 2e-4`, no pit routing:

| | default threads | `OMP_NUM_THREADS=1` |
|---|---|---|
| flow routing | 1.94 ms/step | 1.07 ms/step |
| incision | 1.43 ms/step | 0.80 ms/step |
| diffusion | 4.51 ms/step | 0.21 ms/step |
| **total** | **7.9 ms/step** | **2.1 ms/step** |

With one thread, a 500-step run takes about 1 s. Dataset workers must set `OMP_NUM_THREADS=1` before
importing Landlab. Then 10,000 runs on 32 cores take about 6 minutes.

## Decision: richdem pit filling, in a separate environment

`DepressionFinderAndRouter` would take roughly 1 to 11 hours for 5,000 runs. Without pit routing,
closed hollows in the starting terrain never drain: after 500 steps at `period = 256`, 51% of the
area was in undrained depressions.

`PriorityFloodFlowRouter` is slow for a different reason. `richdem` fills the 128×128 grid in about
1.2 ms, but Landlab's wrapper then loops over every node in Python to find flow directions, about
108 ms per step.

The simulator therefore fills pits with `richdem` on a copy of the terrain, then routes water over
the copy with the normal `FlowAccumulator`. Erosion and diffusion act on the real terrain. This gave
the same result as `PriorityFloodFlowRouter` and takes about 1.9 s per 500-step run.

`richdem` 0.3.4 does not install in the main environment (Python 3.13, numpy 2). It builds on Python
3.10 with numpy 1.26 and one compiler flag. So data generation lives in its own uv project,
`datagen/`, with Landlab 2.9.2.

`pyflwdir` was also tried, because it installs with numpy 2. It was 2 to 3 times slower and gave
different terrain (RMSE 1.8 to 5.0 m against `richdem` after 500 steps), so it was not used.
