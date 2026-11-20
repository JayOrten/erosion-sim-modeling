"""The experiment registry: one entry per experiment in docs/plans/experiments-plan.md.

An entry is data: the models it needs, the evaluation suite that scores them, and the suite's settings.
``runner.py`` trains, evaluates, and submits entries.

Each model is either a ``Config``, which the experiment trains into ``artifacts/models/<experiment>/<label>/``,
or a ``Ref`` to a model another experiment trains. A2's EDM arm, for example, is ``Ref("A1", "base")``:
it is A1's model, trained once. Experiments that use "the best model so far" refer to it through
``DLA_BEST``; change that when the choice changes.
"""

from dataclasses import dataclass, replace

from ..config import PROFILES, Config
from ..dla.suite import AgreementSettings, Settings
from ..erosion.suite import Settings as ErosionSettings
from ..world.suite import Settings as WorldSettings

SUITES = ("dla", "dla_agreement", "erosion", "world")


@dataclass(frozen=True)
class Ref:
    """Model ``label`` of experiment ``experiment``, used here without training it again."""

    experiment: str
    label: str


@dataclass(frozen=True)
class Experiment:
    name: str  # "A1"
    title: str
    models: dict[str, Config | Ref]  # label -> config to train, or another experiment's model; in report order
    suite: str
    settings: object = None  # the suite's settings dataclass; None uses its defaults
    question: str = ""

    def __post_init__(self):
        if self.suite not in SUITES:
            raise ValueError(f"suite must be one of {SUITES}, got {self.suite!r}")
        if not self.models:
            raise ValueError(f"{self.name} has no models")


# ---- Track A: DLA ----

# The DLA base recipe (experiments-plan.md, "DLA base recipe"). The step budget is the phase 1 budget,
# set from A1's timed run.
DLA_BASE = Config(
    profile="dla_base",
    data_profile="dla_master",
    task="dla",
    objective="edm",
    fixed_edges=False,
    # 200,000 runs x 50 = 10 M items, each with its own (t, x), so training rarely sees a pair twice.
    repeats=50,
)

A1_BASE = Ref("A1", "base")
# A2's flow matching arm and A3's DiT arm: the base recipe with one change each.
DLA_FLOW = replace(DLA_BASE, profile="dla_flow", objective="flow_matching")
# DiT S at patch 4: 32 × 32 = 1,024 tokens at 128², about 8.6 M parameters (U-Net: 8.4 M).
DLA_DIT = replace(DLA_BASE, profile="dla_dit", backbone="dit", patch=4)

# The best DLA model so far, used by A5 and A6. A1's model until A2 to A4 are read; then point it at the
# winner, for example Ref("A2", "flow_matching") or Ref("A3", "dit").
DLA_BEST = A1_BASE

EXPERIMENTS: dict[str, Experiment] = {}


def register(*experiments: Experiment) -> None:
    for e in experiments:
        if e.name in EXPERIMENTS:
            raise ValueError(f"experiment {e.name} registered twice")
        EXPERIMENTS[e.name] = e


register(
    Experiment(
        name="A1",
        title="DLA base",
        question="Does the base recipe learn DLA well enough to build on?",
        models={"base": DLA_BASE},
        suite="dla",
    ),
    Experiment(
        name="A2",
        title="Objective",
        question="Does flow matching beat EDM on a random target?",
        models={"edm": A1_BASE, "flow_matching": DLA_FLOW},
        suite="dla",
    ),
    Experiment(
        name="A3",
        title="Backbone",
        question="Does a transformer beat the U-Net on DLA?",
        # Both with A2's winning objective. EDM until A2 is read; then the U-Net arm becomes A2's winner and
        # the DiT arm takes its objective.
        models={"unet": A1_BASE, "dit": DLA_DIT},
        suite="dla",
    ),
    Experiment(
        name="A4",
        title="Sampler steps",
        question="How few sampler steps are enough?",
        models={"edm": A1_BASE, "flow_matching": Ref("A2", "flow_matching")},
        suite="dla",
        settings=Settings(step_sweep=(4, 8, 16, 32, 64)),
    ),
    Experiment(
        name="A5",
        title="Sticking probability",
        question="Does the model follow p, out to the ends of its range?",
        models={"best": DLA_BEST},
        suite="dla",
        settings=Settings(only="p="),
    ),
    Experiment(
        name="A6",
        title="Agreement across steps",
        question="Are samples at different x from one start consistent with each other?",
        models={"best": DLA_BEST},
        suite="dla_agreement",
        settings=AgreementSettings(),
    ),
)

# ---- Track B: single window (experiments-plan.md, "Track B") ----

MAIN_256 = "erosion_master/main_256"
UNIFORM = (("map_shape", "uniform"),)
PIPELINE_CHECK = (("noise", "perlin"), ("fractal", "fbm"), ("warp", 0.0), ("map_shape", "uniform"))

# The track B base recipe: the U-Net with EDM on whole 256² runs, uniform maps, all noise settings. The
# step budget is the phase 1 budget, set from the first timed run.
EROSION_BASE = Config(profile="erosion_base", data_profile=MAIN_256, objective="edm", data=UNIFORM)
B1_BASE = Ref("B1", "base")
# The best single-window recipe so far: B1's model until B2 and B3 are read. Later recipes are built from
# it with replace(...), so set this to the winning Config (for example the B3 DiT config) when chosen,
# and B_BEST_REF to its model.
B_BEST = EROSION_BASE
B_BEST_REF = B1_BASE
# Phase 3: all 4,000 runs (every map shape), trained to convergence.
B_FINAL = replace(B_BEST, profile="erosion_final", data=(), steps=400_000, patience=10)
B_SETTINGS = ErosionSettings(data=UNIFORM)

register(
    Experiment(
        name="B1_pipeline_check",
        title="Pipeline check",
        question="Does the current model work on one fixed noise setup (Perlin, fBm, no warp, uniform map)?",
        models={"pipeline_check": replace(EROSION_BASE, profile="erosion_pipeline_check", data=PIPELINE_CHECK)},
        suite="erosion",
        settings=ErosionSettings(data=PIPELINE_CHECK),
    ),
    Experiment(
        name="B1",
        title="Base",
        question="Does the current model work on varied noise? The reference for every phase 1 experiment.",
        models={"base": EROSION_BASE},
        suite="erosion",
        settings=B_SETTINGS,
    ),
    Experiment(
        name="B2",
        title="Objective",
        question="What does sampling buy over regression, and does the diffusion formulation matter?",
        models={"edm": B1_BASE,
                "regression": replace(EROSION_BASE, profile="erosion_regression", objective="regression"),
                "flow_matching": replace(EROSION_BASE, profile="erosion_flow", objective="flow_matching")},
        suite="erosion",
        settings=B_SETTINGS,
    ),
    Experiment(
        name="B3",
        title="Backbone",
        question="Does attention over the whole window help rivers?",
        models={"unet_s": B_BEST_REF,
                "unet_l": replace(B_BEST, profile="erosion_unet_l", backbone="unet", channels=(64, 128, 256, 512)),
                "dit_s": replace(B_BEST, profile="erosion_dit_s", backbone="dit", patch=8),
                "dit_l": replace(B_BEST, profile="erosion_dit_l", backbone="dit", patch=8, dit_depth=12, dit_width=384,
                                 dit_heads=8),
                "fno": replace(B_BEST, profile="erosion_fno", backbone="fno", objective="regression")},
        suite="erosion",
        settings=B_SETTINGS,
    ),
    Experiment(
        name="B4",
        title="Data scaling",
        question="Does more data still help?",
        models={f"runs_{n}": replace(B_BEST, profile=f"erosion_runs_{n}", max_runs=n) for n in (333, 667, 1_333)}
        | {"runs_2667": B_BEST_REF},
        suite="erosion",
        settings=B_SETTINGS,
    ),
    Experiment(
        name="B5",
        title="Sampler steps",
        question="How few steps are enough?",
        models={"edm": B1_BASE, "flow_matching": Ref("B2", "flow_matching"), "regression": Ref("B2", "regression")},
        suite="erosion",
        settings=ErosionSettings(data=UNIFORM, step_sweep=(4, 8, 16, 32, 64), samples=1),
    ),
    Experiment(
        name="B6",
        title="Erodibility maps",
        question="Does the model need to see the map, and do hard bands make the simulator's ridges and steps?",
        models={"map_off": replace(B_BEST, profile="erosion_all_maps", data=()),
                "map_on": replace(B_BEST, profile="erosion_map_input", data=(), inputs=("map",))},
        suite="erosion",
        settings=ErosionSettings(),  # every test run, broken down by map shape and contrast
    ),
    Experiment(
        name="B7",
        title="Hydrology inputs and losses",
        question="Which change makes rivers most correct? Pit filling is scored for every model.",
        models={"base": B_BEST_REF,
                "drainage_input": replace(B_BEST, profile="erosion_drainage_input", inputs=("drainage",)),
                "loss_0.25": replace(B_BEST, profile="erosion_drainage_loss_025", drainage_weight=0.25),
                "loss_0.5": replace(B_BEST, profile="erosion_drainage_loss_05", drainage_weight=0.5)},
        suite="erosion",
        settings=ErosionSettings(data=UNIFORM, fill_pits=True),
    ),
    Experiment(
        name="B8",
        title="Residual diffusion",
        question="Does splitting the change into a regression mean and a sampled residual help?",
        models={"regression": Ref("B2", "regression"),
                "residual": replace(B_BEST, profile="erosion_residual", objective="edm", residual_of="B2/regression")},
        suite="erosion",
        settings=ErosionSettings(data=UNIFORM, step_sweep=(4, 8, 16, 32), samples=1),
    ),
    Experiment(
        name="B9",
        title="Final single-window model",
        question="The headline track B numbers, over three seeds. Ablations are added once phase 2 is read.",
        models={f"seed{k}": replace(B_FINAL, seed=k) for k in range(3)},
        suite="erosion",
        settings=ErosionSettings(),
    ),
    Experiment(
        name="B10_fractal",
        title="Generalization: held-out fractal type",
        question="Trained on fBm only, how does the model do on ridged terrain, against B9?",
        models={"final": Ref("B9", "seed0"),
                "fbm_only": replace(B_FINAL, profile="erosion_fbm_only", data=(("fractal", "fbm"),))},
        suite="erosion",
        settings=ErosionSettings(data=(("fractal", "ridged"),)),
    ),
    Experiment(
        name="B10_ood",
        title="Generalization: parameters out of range",
        question="How does the final model do with one erosion parameter above the training range?",
        models={"final": Ref("B9", "seed0")},
        suite="erosion",
        settings=ErosionSettings(data_profile="erosion_master/ood_256"),
    ),
    Experiment(
        name="B10_long",
        title="Generalization: elapsed times beyond training",
        question="Up to 150,000 yr, in one call and chained, from the start terrain.",
        models={"final": Ref("B9", "seed0")},
        suite="erosion",
        settings=ErosionSettings(data_profile="erosion_master/long_256", ends_yr=(50_000.0, 100_000.0, 150_000.0),
                                 chains=((3, 50_000.0), (6, 25_000.0), (15, 10_000.0)), samples=1),
    ),
    Experiment(
        name="B11",
        title="Parameters and time scales",
        question="Where in parameter space is the model good or bad? (Read the breakdowns.)",
        models={"final": Ref("B9", "seed0")},
        suite="erosion",
        settings=ErosionSettings(pairs_per_run=8),
    ),
    Experiment(
        name="B12",
        title="Baselines",
        question="How does the final model compare with regression, the analytical method, and no change?",
        models={"final": Ref("B9", "seed0"),
                "regression": replace(B_FINAL, profile="erosion_final_regression", objective="regression")},
        suite="erosion",
        settings=ErosionSettings(),
    ),
    Experiment(
        name="B13",
        title="Speed",
        question="How fast can the final model be, and what does speed cost? Students need an EDM final model.",
        models={"final": Ref("B9", "seed0"),
                "student": replace(B_FINAL, profile="erosion_student", objective="consistency", teacher="B9/seed0",
                                   steps=50_000, patience=None)},
        suite="erosion",
        settings=ErosionSettings(step_sweep=(1, 2, 4, 8, 32), samples=1),
    ),
)

# ---- Track C: infinite world (experiments-plan.md, "Track C") ----

MAIN_512 = "erosion_master/main_512"
# A world level: windows of interior crops, 32 cells from the fixed edges, flipped and rotated at random.
LEVEL = replace(B_BEST, profile="level", data_profile=MAIN_512, fixed_edges=False, augment=True, margin=32)
COARSE = replace(LEVEL, profile="coarse", cell=4, window=64)
FINE = replace(LEVEL, profile="fine", cell=1, window=128, context_cell=4)
C1_WORLD = (("world", ("coarse", "fine")),)
C_SETTINGS = WorldSettings(worlds=C1_WORLD, data=UNIFORM)


RUN_SIZES = {"main_256": 256, "main_512": 512, "main_1024": 1024, "main_2048": 2048}


def _stack(prefix: str, group: str, cells_windows: tuple[tuple[int, int], ...], **changes) -> dict[str, Config]:
    """Levels of a world, coarsest first: ``(cell, window)`` per level, each reading its parent's change.
    Each level keeps the 32-cell edge margin if its window fits with it, else the largest margin that fits
    (0 when the window is the whole run, as for band 2's 16 m level)."""
    size = RUN_SIZES[group.split("/")[1]]
    out, parent = {}, None
    for cell, window in cells_windows:
        align = parent or cell
        margin = max(0, min(32, (size - window * cell) // 2)) // align * align
        out[f"{prefix}_{cell}m"] = replace(LEVEL, profile=f"{prefix}_{cell}m", data_profile=group, cell=cell,
                                          window=window, context_cell=parent, margin=margin, **changes)
        parent = cell
    return out


register(
    Experiment(
        name="C1_pipeline_check",
        title="Base world: pipeline check",
        question="Does the two-level world work on one fixed noise setup?",
        models={"coarse": replace(COARSE, profile="coarse_pipeline_check", data=PIPELINE_CHECK),
                "fine": replace(FINE, profile="fine_pipeline_check", data=PIPELINE_CHECK)},
        suite="world",
        settings=WorldSettings(worlds=C1_WORLD, data=PIPELINE_CHECK),
    ),
    Experiment(
        name="C1",
        title="Base world",
        question="Does the two-level world work at band 1 on varied noise? The track C reference.",
        models={"coarse": replace(COARSE, data=UNIFORM), "fine": replace(FINE, data=UNIFORM)},
        suite="world",
        settings=C_SETTINGS,
    ),
    Experiment(
        name="C2",
        title="Edges",
        question="Does training on whole closed runs teach the levels fixed edges the world does not have?",
        models={"coarse": Ref("C1", "coarse"), "fine": Ref("C1", "fine"),
                # A 256² closed run is exactly one 64-cell coarse window; fine windows anywhere, edges included.
                "coarse_closed": replace(COARSE, profile="coarse_closed", data_profile=MAIN_256, data=UNIFORM, margin=0),
                "fine_closed": replace(FINE, profile="fine_closed", data_profile=MAIN_256, data=UNIFORM, margin=0)},
        suite="world",
        settings=WorldSettings(worlds=(("interior", ("coarse", "fine")), ("closed", ("coarse_closed", "fine_closed"))),
                               data_profile=MAIN_512, data=UNIFORM),
    ),
    Experiment(
        name="C3",
        title="Sampler settings",
        question="Which steps and blend phases give the best world? One setting changed at a time from C1's.",
        models={"coarse": Ref("C1", "coarse"), "fine": Ref("C1", "fine")},
        suite="world",
        settings=WorldSettings(data=UNIFORM, worlds=tuple(
            [(f"steps_{n}", ("coarse", "fine"), (("steps", n),)) for n in (8, 16, 32, 64)]
            + [("T1", ("coarse", "fine"), (("splits", ()),))]
            + [(f"T2_split_{k}", ("coarse", "fine"), (("splits", (float(k),)),)) for k in (2, 10)]
            + [(f"T3_{a}_{b}", ("coarse", "fine"), (("splits", (float(a), float(b))),)) for a, b in ((10, 2), (20, 5))])),
    ),
    Experiment(
        name="C4",
        title="Coarse conditioning noise",
        question="Does the fine level do better when trained not to trust its coarse input exactly?",
        models={"coarse": Ref("C1", "coarse"), "fine_0": Ref("C1", "fine")}
        | {f"fine_{k}": replace(FINE, profile=f"fine_context_noise_{k}", data=UNIFORM, context_noise=k)
           for k in (0.05, 0.1, 0.2, 0.4)},
        suite="world",
        settings=WorldSettings(data=UNIFORM, worlds=tuple((f"noise_{k}", ("coarse", f"fine_{k}"))
                                                          for k in (0, 0.05, 0.1, 0.2, 0.4))),
    ),
    Experiment(
        name="C5_band2",
        title="Scale: band 2",
        question="Band 2 (period 64 to 128): two levels with a 128-cell coarse window, or three levels?",
        models=_stack("two", "erosion_master/main_1024", ((4, 128), (1, 128)), data=UNIFORM)
        | _stack("three", "erosion_master/main_1024", ((16, 64), (4, 64), (1, 128)), data=UNIFORM),
        suite="world",
        settings=WorldSettings(data=UNIFORM, worlds=(("two_levels", ("two_4m", "two_1m")),
                                                     ("three_levels", ("three_16m", "three_4m", "three_1m"))), runs=4),
    ),
    Experiment(
        name="C5_band3",
        title="Scale: band 3",
        question="Band 3 (period 128 to 256): three levels or four? The 64 m level's window is 16 cells, not "
                 "the plan's 64, which would not fit a 2048² run.",
        models=_stack("three", "erosion_master/main_2048", ((16, 64), (4, 64), (1, 128)), data=UNIFORM)
        | _stack("four", "erosion_master/main_2048", ((64, 16), (16, 64), (4, 64), (1, 128)), data=UNIFORM),
        suite="world",
        settings=WorldSettings(data=UNIFORM, margin=64, worlds=(("three_levels", ("three_16m", "three_4m", "three_1m")),
                                                                ("four_levels", ("four_64m", "four_16m", "four_4m",
                                                                                 "four_1m"))), runs=2),
    ),
    Experiment(
        name="C6",
        title="Final world",
        question="The headline track C numbers at band 1, over three seeds. Bands 2 and 3 follow C5's choice.",
        models={f"{level}_seed{k}": replace(cfg, profile=f"final_{level}", data=(), seed=k, steps=400_000, patience=10)
                for k in range(3) for level, cfg in (("coarse", COARSE), ("fine", FINE))},
        suite="world",
        settings=WorldSettings(worlds=tuple((f"seed{k}", (f"coarse_seed{k}", f"fine_seed{k}")) for k in range(3))),
    ),
    Experiment(
        name="C7_ood",
        title="Generalization at world scale: parameters out of range",
        question="Do B10's out-of-range results hold in the world?",
        models={"coarse": Ref("C6", "coarse_seed0"), "fine": Ref("C6", "fine_seed0")},
        suite="world",
        settings=WorldSettings(worlds=C1_WORLD, data_profile="erosion_master/ood_512"),
    ),
    Experiment(
        name="C7_long",
        title="Generalization at world scale: long elapsed times",
        question="Up to 150,000 yr in one world query. (Chained world calls are not built.)",
        models={"coarse": Ref("C6", "coarse_seed0"), "fine": Ref("C6", "fine_seed0")},
        suite="world",
        settings=WorldSettings(worlds=C1_WORLD, data_profile="erosion_master/long_512",
                               ends_yr=(50_000.0, 100_000.0, 150_000.0)),
    ),
    Experiment(
        name="C8",
        title="Baselines",
        question="Does the world beat every way of making a bounded method infinite? Needs `erosion-datagen "
                 "chunked main_512 test --chunk 256` for the simulator chunks.",
        models={"coarse": Ref("C6", "coarse_seed0"), "fine": Ref("C6", "fine_seed0"), "single_window": Ref("B9", "seed0")},
        suite="world",
        settings=WorldSettings(worlds=C1_WORLD, chunk_model="single_window", runs=None,
                               baselines=("analytical", "analytical_chunks", "analytical_blended", "model_chunks",
                                          "simulator_chunks")),
    ),
    Experiment(
        name="C9",
        title="Speed",
        question="Is the world fast enough to explore? The final world against its distilled students.",
        models={"coarse": Ref("C6", "coarse_seed0"), "fine": Ref("C6", "fine_seed0")}
        | {f"{level}_student": replace(cfg, profile=f"{level}_student", data=(), objective="consistency",
                                       teacher=f"C6/{level}_seed0", steps=50_000)
           for level, cfg in (("coarse", COARSE), ("fine", FINE))},
        suite="world",
        settings=WorldSettings(worlds=(("final", ("coarse", "fine")),
                                       ("students_1_step", ("coarse_student", "fine_student"), (("steps", 1),)),
                                       ("students_2_steps", ("coarse_student", "fine_student"), (("steps", 2),))),
                               runs=4, baselines=()),
    ),
)

# ---- Smoke experiments: seconds on CPU, for tests and for checking a setup end to end ----

register(
    Experiment(
        name="smoke_A",
        title="DLA smoke",
        models={"base": PROFILES["smoke_dla"]},
        suite="dla",
        settings=Settings(steps=2, batch_size=8),
    ),
    Experiment(
        name="smoke_A_steps",
        title="DLA objectives and step sweep smoke",
        models={"edm": Ref("smoke_A", "base"),
                "flow_matching": replace(PROFILES["smoke_dla"], profile="smoke_dla_flow", objective="flow_matching")},
        suite="dla",
        settings=Settings(batch_size=8, step_sweep=(1, 2)),
    ),
    Experiment(
        name="smoke_B",
        title="Single-window smoke",
        models={"edm": replace(PROFILES["smoke"], profile="smoke_erosion", data_profile="erosion_smoke/smoke",
                               objective="edm"),
                "regression": replace(PROFILES["smoke"], profile="smoke_erosion_regression",
                                      data_profile="erosion_smoke/smoke")},
        suite="erosion",
        settings=ErosionSettings(pairs_per_run=2, steps=2, batch_size=8, fill_pits=True),
    ),
    Experiment(
        name="smoke_C",
        title="World smoke",
        models={"coarse": replace(PROFILES["smoke"], profile="smoke_coarse", data_profile="erosion_smoke/smoke",
                                  objective="edm", fixed_edges=False, augment=True, cell=2, window=8, margin=4),
                "fine": replace(PROFILES["smoke"], profile="smoke_fine", data_profile="erosion_smoke/smoke",
                                objective="edm", fixed_edges=False, augment=True, cell=1, window=16, context_cell=2,
                                margin=4),
                "single_window": Ref("smoke_B", "edm")},
        suite="world",
        settings=WorldSettings(worlds=(("world", ("coarse", "fine")),), runs=2, ends_yr=(1_000.0, 2_000.0), margin=4,
                               steps=2, chunk=16, chunk_edge=2, chunk_model="single_window", level_pairs_per_run=2,
                               seam_windows=2, baselines=("analytical", "analytical_chunks", "analytical_blended",
                                                          "model_chunks", "simulator_chunks")),
    ),
    Experiment(
        name="smoke_A_agreement",
        title="DLA agreement smoke",
        models={"base": Ref("smoke_A", "base")},
        suite="dla_agreement",
        settings=AgreementSettings(steps=2, batch_size=8, fresh_starts=3),
    ),
)


def get(name: str) -> Experiment:
    if name not in EXPERIMENTS:
        raise KeyError(f"no experiment {name!r}; registered: {', '.join(EXPERIMENTS)}")
    return EXPERIMENTS[name]


def resolve(name: str, label: str) -> tuple[str, str, Config]:
    """The experiment and label that own model ``label`` of experiment ``name``, and its config. Follows
    ``Ref`` entries to the experiment that trains the model."""
    seen = set()
    while True:
        exp = get(name)
        if label not in exp.models:
            raise KeyError(f"{name} has no model {label!r}; models: {', '.join(exp.models)}")
        entry = exp.models[label]
        if isinstance(entry, Config):
            return name, label, entry
        if (name, label) in seen:
            raise ValueError(f"models refer to each other in a loop at {name}/{label}")
        seen.add((name, label))
        name, label = entry.experiment, entry.label


def _check_refs() -> None:
    for name, exp in EXPERIMENTS.items():
        for label in exp.models:
            resolve(name, label)


_check_refs()
