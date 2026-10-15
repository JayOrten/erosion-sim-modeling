# AGENTS.md

Guidance for coding agents working in this repository.

## What this project is

A research prototype for a neural surrogate of terrain erosion. A surrogate is a learned model that
approximates a slow simulator. The model takes a terrain heightmap plus erosion parameters (for
example rainfall or erodibility) and outputs the eroded terrain, without running the simulation.

The repository is new. Nothing is built yet. Most decisions below are still open; do not treat them
as settled.

## Where it comes from

This project continues ideas from the sibling repository `../seeded-pa-vae`. That repository trains
VAEs (variational autoencoders) that expose a stateless random-access interface:

```python
f(seed, query, controls) -> value
```

Queries can arrive in any order and receive no earlier outputs, yet all answers for one seed must
describe one coherent sampled object. It was tested first on preferential-attachment trees (`src/svae/`)
and then on diffusion-limited aggregation (DLA) trees (`src/dla/`). Read `../seeded-pa-vae/docs/` for
the research framing, especially `research-question.md` and `architecture.md`.

The long-term link to erosion is chunked terrain, as in Minecraft: generate any chunk on demand from a
seed, and have neighboring chunks agree with each other. That work used a split between a global
latent (shared by the whole world) and local latents (per region). Here, the first step is smaller: a
single-chunk surrogate.

## Research context

### Novelty boundary

As of the last literature search, no learned surrogate was found that emulates a hydraulic erosion
simulator with parameter conditioning. The general recipe (a diffusion model or neural operator
conditioned on simulation parameters) is well established elsewhere. So:

- The single-chunk model is a direct application of known methods to a new domain. It is a first
  step, not the contribution.
- A likely contribution is hydrological correctness as an evaluation target, and consistency across
  chunks.
- Do not describe the method itself as novel in docs or code comments.

### Closest prior work

- Guérin et al., SIGGRAPH Asia 2017. Conditional GANs (generative adversarial networks) for terrain
  authoring, including an erosion step. Image to image, no parameter control.
- Neural operators for landscape evolution, Earth Surface Dynamics 2025. Fourier Neural Operators
  trained on stream power erosion runs. Deterministic, geological scale.
- Postfire debris-flow surrogate, 2025. A U-Net taking a 256×256 terrain patch plus six simulation
  parameters, with parameters sampled by Latin Hypercube Sampling and tested on unseen terrain. This
  is the best template for the data design.
- Shokar et al., 2025. A parameter-conditioned emulator for stochastic spatio-temporal systems.
  Injects parameters with adaptive layer normalization (AdaLN).
- ACM-FD, ICML 2025. One diffusion surrogate for forward prediction and inverse problems.
- Chandra et al., 2020. A surrogate for the Badlands landscape evolution model. It approximates the
  likelihood for parameter inference, not the terrain itself.

### Main competition: fast non-learned erosion

Reviewers will ask why a learned surrogate beats these. Plan to include at least one as a baseline.

- Tzathas, Gailleton, Steer & Cordonnier, Eurographics 2024. Analytical solutions of the stream power
  law. Erosion time becomes a function parameter instead of an iteration count.
- Schott et al., SIGGRAPH 2024. Fast approximations of thermal, stream power, and hillslope erosion
  at several scales, used to amplify low-resolution terrain.
- McDonald & Cordonnier, SIGGRAPH 2026, "Stochastic geomorphological transport for terrain erosion
  simulation". Contents not yet checked.

The honest case for a learned surrogate: it can emulate any simulator, including particle-based
hydraulic erosion that has no analytical shortcut; it gives one interface across processes; and it is
differentiable.

### Open design decisions

Ask the user before settling any of these.

- Which erosion simulator generates the training data.
- Model family: diffusion model, U-Net, or neural operator.
- Source of input terrain: synthetic, real elevation data, or both.
- Ground truth for a chunk: the simulator run on that chunk alone, or a crop from a run on a larger
  domain. The suggested start is the first, then measure the gap to the second.
- Evaluation metrics, beyond pixel error. Hydrological checks (drainage networks, flow accumulation,
  absence of spurious pits) are a likely focus.

## Conventions

Follow the patterns in `../seeded-pa-vae` unless there is a reason not to. Copy code and adapt it
rather than importing across repositories.

- **Environment.** Use `uv`. Install with `uv sync`; run with `uv run`. Commit `uv.lock`.
- **Layout.** Source in `src/<package>/`, installed editably via setuptools. Tests in `tests/`.
  Notebooks in `notebooks/`. Research notes and plans in `docs/`.
- **Modules.** Split by role: `simulate.py`, `config.py`, `train.py`, `generate.py`, `evaluate.py`.
- **Configs.** A frozen dataclass `Config` plus a `PROFILES` dict. Always include a tiny `smoke`
  profile that runs in seconds on CPU, then larger ones.
- **Simulators.** Write custom simulators in Numba when existing packages do not fit. Record why an
  existing package was rejected in `docs/`. Numba random state is per thread, so build datasets with
  one seed per process.
- **Seeding.** Derive random streams from a public integer seed by hashing a canonical JSON record
  with SHA-256 and seeding NumPy's PCG64. Give each role (latent, per-query noise) its own stream.
- **Checkpoints.** Save to `artifacts/`, which is gitignored. Prefix filenames by model so runs do
  not overwrite each other.
- **Evaluation.** Experiment functions return their results. Plotting and printing are on by default;
  tests pass `show=False`.
- **Baselines.** Every model is compared against the simulator and at least one simple baseline.
- **Tests.** `uv run pytest`. Tests use the `smoke` profile.

## How to work

- One focused task at a time. Small steps, with a check-in after each one.
- Do not spawn subagents or parallel work unless asked.
- Plans go in `docs/` as numbered steps. Each step says in plain words what gets built or changed,
  then the details, and ends with a check-in.
- When it is unclear whether to edit files directly, show the code and ask first.
- Never run `sudo`. Give the command to the user instead.
- Do not commit unless asked.

## Writing style

Applies to docs, comments, and messages.

- Short sentences, one idea each, common words.
- Say what something is, then what it does.
- Expand every acronym and jargon term on first use.
- Use the names the field actually uses. Do not coin terms.
- No filler, no enthusiasm, no emojis.
