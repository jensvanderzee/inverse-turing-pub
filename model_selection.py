# -*- coding: utf-8 -*-
"""
Picking the best real-data model from held-out test metrics.

A model that diverged on any held-out site cannot be best. Its MSE there is NaN
(written to test_metrics.csv as an empty field) or inf, and a NaN-skipping mean
would rank it on the sites it happened to survive instead.
"""
import numpy as np
import pandas as pd


def mean_test_mse(metrics: pd.DataFrame) -> pd.Series:
    """Mean held-out MSE per model_id; NaN if the model diverged on any site."""
    mse = metrics["mse"].replace([np.inf, -np.inf], np.nan)
    return mse.groupby(metrics["model_id"]).apply(lambda v: v.mean(skipna=False))


def best_model_by_test_mse(metrics: pd.DataFrame) -> tuple:
    """(model_id, mean MSE) of the model with the lowest mean held-out MSE."""
    mean_mse = mean_test_mse(metrics)
    usable = mean_mse.dropna()
    if usable.empty:
        raise ValueError("every model diverged on at least one held-out site")
    n_diverged = len(mean_mse) - len(usable)
    if n_diverged:
        print(f"{n_diverged} of {len(mean_mse)} models diverged on at least one "
              f"held-out site and cannot be best")
    best_id = usable.idxmin()
    return int(best_id), float(usable[best_id])
