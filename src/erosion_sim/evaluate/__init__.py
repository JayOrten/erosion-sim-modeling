"""Evaluate trained models against the simulator.

- ``pairs``: predict a fixed sample of pairs with one or more models (``predict_pairs``).
- ``metrics``: accuracy metrics on that output: height error, rivers, spectrum, where models fail.
- ``rollouts``: error as model calls are chained over time.
- ``sweeps``: metrics against the number of diffusion sampler steps.
- ``speed``: time the models and the simulator.
- ``plots``: pictures and animations of models next to the simulator.

Everything is importable from here, for example ``from erosion_sim.evaluate import predict_pairs``.
"""

from .metrics import (error_by_condition, error_by_elapsed, error_metrics, fine_detail, hydrology_metrics,
                      radial_power_spectrum, sample_spread, spectrum_comparison)
from .pairs import FACTORS, add_method, make_pairs, model_methods, per_pair_skill, predict_pairs, subset_pairs
from .plots import comparison_gif, show_comparison, show_predictions
from .rollouts import rollout_error
from .speed import calls_needed, speed_comparison, time_model, time_simulator
from .sweeps import step_sweep

__all__ = [
    "FACTORS", "add_method", "calls_needed", "comparison_gif", "error_by_condition", "error_by_elapsed",
    "error_metrics", "fine_detail", "hydrology_metrics", "make_pairs", "model_methods", "per_pair_skill", "predict_pairs",
    "radial_power_spectrum", "rollout_error", "sample_spread", "show_comparison", "show_predictions",
    "spectrum_comparison", "speed_comparison", "step_sweep", "subset_pairs", "time_model", "time_simulator",
]
