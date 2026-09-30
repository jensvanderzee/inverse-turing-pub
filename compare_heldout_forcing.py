# -*- coding: utf-8 -*-
"""
How the held-out scores change with the rainfall and step-size fixes.

Scores every fitted model on the held-out sites under three setups:

    published    raw weekly CSVs, 4 steps/week   (what test_metrics.csv was made with)
    rain_fixed   checked weekly forcing, 4 steps/week
    both_fixed   checked weekly forcing, the steps/week the models were fitted at

so the effect of each fix can be read off separately. "Checked" forcing is what
the loaders now do: an incomplete weekly record (all of subsite_k, subsite_j from
2020) is replaced by the annual total spread evenly across 52 weeks, or by the
re-downloaded weekly series once download_missing_weekly_precip.py has been run.

Writes to results/real_data/heldout_forcing_comparison/:
    metrics_long.csv   one row per setup x model x site
    summary.csv        per setup and site: median/mean MSE, divergent runs,
                       zero-change baseline, best model
    mse_published_vs_fixed.png

Run from the repo root:
    python compare_heldout_forcing.py [--params <csv>]
"""
#%%
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from realdata_test_invPDE import (
    DATA_DIR, DEVICE, NDVI_TO_BIOMASS_MULTIPLIER, PARAM_CSV, TEST_SITES,
    STEPS_PER_WEEK as FIT_STEPS_PER_WEEK,
    build_model_from_row, evaluate_model_on_site, load_parameters,
)
from realdata_train_invPDE import RealDataLoader

PUBLISHED_STEPS_PER_WEEK = 4
SAVE_DIR = "results/real_data/heldout_forcing_comparison"


def load_test_sites(check_weekly: bool) -> dict:
    loader = RealDataLoader(DATA_DIR, selected_sites=TEST_SITES, device=DEVICE,
                            use_weekly_precip=True, check_weekly=check_weekly)
    data = loader.get_training_data(ndvi_to_biomass_multiplier=NDVI_TO_BIOMASS_MULTIPLIER)
    return data["location_time_series"]


def zero_change_mse(time_series: list) -> float:
    """Delta-MSE of predicting no change at all: the score any model has to beat."""
    return float(np.mean([
        torch.mean((b["biomass"] - a["biomass"]) ** 2).item()
        for a, b in zip(time_series[:-1], time_series[1:])
    ]))


def score_all(param_df: pd.DataFrame, sites: dict, steps_per_week: int, setup: str) -> list:
    rows = []
    for model_id, row in param_df.iterrows():
        model = build_model_from_row(row)
        for site_name, ts in sites.items():
            m = evaluate_model_on_site(model, ts, steps_per_week=steps_per_week)
            rows.append({"setup": setup, "model_id": model_id, "site": site_name, **m})
        print(f"  {setup}: model {model_id} done", flush=True)
    return rows


def summarise(long_df: pd.DataFrame, baselines: dict) -> pd.DataFrame:
    rows = []
    for (setup, site), g in long_df.groupby(["setup", "site"], sort=False):
        finite = g[np.isfinite(g["mse"])]
        best = finite.loc[finite["mse"].idxmin()] if len(finite) else None
        rows.append({
            "setup": setup,
            "site": site,
            "median_mse": finite["mse"].median(),
            "mean_mse": finite["mse"].mean(),
            "median_corr": finite["correlation"].median(),
            "n_models": len(g),
            "n_divergent": int((~np.isfinite(g["mse"])).sum()),
            "zero_change_mse": baselines[site],
            "n_beating_zero_change": int((finite["mse"] < baselines[site]).sum()),
            "best_model_on_site": None if best is None else best["model_id"],
        })
    return pd.DataFrame(rows)


def mean_mse_per_model(long_df: pd.DataFrame) -> pd.DataFrame:
    """Mean MSE across sites, one column per setup. A model that diverges on any
    site gets NaN: skipping the NaN would rank it on the sites it survived."""
    return (long_df.replace([np.inf, -np.inf], np.nan)
            .groupby(["setup", "model_id"], sort=False)["mse"]
            .apply(lambda v: v.mean(skipna=False))
            .unstack("setup"))


def plot_published_vs_fixed(long_df: pd.DataFrame, path: str):
    wide = long_df.pivot_table(index=["model_id", "site"], columns="setup", values="mse").reset_index()
    sites = list(dict.fromkeys(long_df["site"]))
    fig, axes = plt.subplots(1, len(sites), figsize=(5 * len(sites), 4.6), squeeze=False)
    for ax, site in zip(axes[0], sites):
        w = wide[(wide["site"] == site)].replace([np.inf, -np.inf], np.nan).dropna()
        ax.scatter(w["published"], w["both_fixed"], s=18, alpha=0.7)
        if len(w):
            lo = min(w["published"].min(), w["both_fixed"].min())
            hi = max(w["published"].max(), w["both_fixed"].max())
            ax.plot([lo, hi], [lo, hi], "k--", linewidth=1)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(site)
        ax.set_xlabel("MSE, published setup")
        ax.set_ylabel("MSE, both fixes")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--params", default=PARAM_CSV, help="fitted parameter table")
    args = parser.parse_args()
    os.makedirs(SAVE_DIR, exist_ok=True)

    param_df = load_parameters(args.params)
    fit_spw = FIT_STEPS_PER_WEEK
    print(f"{len(param_df)} models; fitted at {fit_spw} steps/week\n")

    raw_sites = load_test_sites(check_weekly=False)
    checked_sites = load_test_sites(check_weekly=True)
    baselines = {name: zero_change_mse(ts) for name, ts in checked_sites.items()}

    setups = [
        ("published", raw_sites, PUBLISHED_STEPS_PER_WEEK),
        ("rain_fixed", checked_sites, PUBLISHED_STEPS_PER_WEEK),
        ("both_fixed", checked_sites, fit_spw),
    ]
    rows = []
    for setup, sites, spw in setups:
        rows += score_all(param_df, sites, spw, setup)
    long_df = pd.DataFrame(rows)
    long_df.to_csv(os.path.join(SAVE_DIR, "metrics_long.csv"), index=False)

    summary = summarise(long_df, baselines)
    summary.to_csv(os.path.join(SAVE_DIR, "summary.csv"), index=False)

    pd.set_option("display.width", 160)
    print("\nPer site (median over models; divergent runs excluded):")
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.4g}"))

    per_model = mean_mse_per_model(long_df)
    print("\nBest model by mean MSE across held-out sites (models that diverge anywhere excluded):")
    for setup in per_model.columns:
        col = per_model[setup].dropna()
        print(f"  {setup:<11} model {col.idxmin()}  (mean MSE {col.min():.4g}; "
              f"{per_model[setup].isna().sum()} models excluded)")

    # Does the ranking of models survive the fixes?
    both = per_model[["published", "both_fixed"]].dropna()
    rho = both["published"].corr(both["both_fixed"], method="spearman")
    print(f"\nSpearman rank correlation of mean MSE, published vs both fixed "
          f"({len(both)} models finite in both): {rho:.3f}")

    plot_published_vs_fixed(long_df, os.path.join(SAVE_DIR, "mse_published_vs_fixed.png"))
    print(f"\nOutputs in {SAVE_DIR}/")


if __name__ == "__main__":
    main()

# %%
