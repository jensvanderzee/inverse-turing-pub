# -*- coding: utf-8 -*-
"""
Ground-truth coefficients for the synthetic experiments.

The 1-site and 4-site experiments were generated with *different* coefficients
and a different model-time length for one calendar year, so each regime must be
scored against its own truth. The training scripts build their ground-truth model
from these dicts and record them in every result JSON; the comparison scripts read
them back with `ground_truth_for_run`, falling back to the dicts here for result
files written before the field existed.
"""

# train_invPDE_synthetic_batch.py
GROUND_TRUTH_4SITE = {
    "infiltration_rate":             0.1,
    "seepage_rate":                  0.4,
    "plant_uptake_rate":             0.15,
    "mortality_rate":                0.3,
    "evaporation_rate":              0.3,
    "water_use_efficiency":          0.35,
    "surface_water_diffusion_coeff": 8.0,
    "soil_water_diffusion_coeff":    1.0,
    "biomass_diffusion_coeff":       0.05,
}
YEAR_TIME_UNITS_4SITE = 1.5

# train_invPDE_synthetic_batch_1site.py
GROUND_TRUTH_1SITE = {
    "infiltration_rate":             0.2,
    "seepage_rate":                  0.8,
    "plant_uptake_rate":             0.35,
    "mortality_rate":                0.6,
    "evaporation_rate":              0.6,
    "water_use_efficiency":          0.35,
    "surface_water_diffusion_coeff": 8.0,
    "soil_water_diffusion_coeff":    1.0,
    "biomass_diffusion_coeff":       0.05,
}
YEAR_TIME_UNITS_1SITE = 1.0

PARAM_NAMES = list(GROUND_TRUTH_4SITE.keys())


def ground_truth_for_run(run: dict, default: dict) -> dict:
    """Ground truth recorded in a result JSON, or `default` if it predates the field."""
    return run.get("ground_truth", default)
