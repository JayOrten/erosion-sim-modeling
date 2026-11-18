"""The experiment registry: one entry per experiment in docs/plans/experiments-plan.md.

An entry is data: the models it needs (label -> ``Config``), the evaluation suite that scores them, and
the suite's settings. ``runner.py`` trains, evaluates, and submits entries.

A model that appears in several experiments is the same ``Config`` object or an equal one, so it has
the same hash and is trained once (``identity.py``). Experiments that use "the best model so far" name
that model directly; change the entry when the choice changes.
"""

from dataclasses import dataclass

from .config import PROFILES, Config
from .dla.suite import AgreementSettings, Settings

SUITES = ("dla", "dla_agreement")


@dataclass(frozen=True)
class Experiment:
    name: str  # "A1"
    title: str
    models: dict[str, Config]  # label -> config, in report order
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
        name="A5",
        title="Sticking probability",
        question="Does the model follow p, out to the ends of its range?",
        models={"best": DLA_BASE},  # the best of A1 to A4; A1's model until A2 to A4 exist
        suite="dla",
        settings=Settings(only="p="),
    ),
    Experiment(
        name="A6",
        title="Agreement across steps",
        question="Are samples at different x from one start consistent with each other?",
        models={"best": DLA_BASE},  # the best of A1 to A4
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
        name="smoke_A_agreement",
        title="DLA agreement smoke",
        models={"base": PROFILES["smoke_dla"]},
        suite="dla_agreement",
        settings=AgreementSettings(steps=2, batch_size=8, fresh_starts=3),
    ),
)


def get(name: str) -> Experiment:
    if name not in EXPERIMENTS:
        raise KeyError(f"no experiment {name!r}; registered: {', '.join(EXPERIMENTS)}")
    return EXPERIMENTS[name]
