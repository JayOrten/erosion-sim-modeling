"""The experiment registry: one entry per experiment in docs/plans/experiments-plan.md.

An entry is data: the models it needs, the evaluation suite that scores them, and the suite's settings.
``runner.py`` trains, evaluates, and submits entries.

Each model is either a ``Config``, which the experiment trains into ``artifacts/models/<experiment>/<label>/``,
or a ``Ref`` to a model another experiment trains. A2's EDM arm, for example, is ``Ref("A1", "base")``:
it is A1's model, trained once. Experiments that use "the best model so far" refer to it through
``DLA_BEST``; change that when the choice changes.
"""

from dataclasses import dataclass, replace

from .config import PROFILES, Config
from .dla.suite import AgreementSettings, Settings

SUITES = ("dla", "dla_agreement")


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
