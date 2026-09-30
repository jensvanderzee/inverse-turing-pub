"""Compare learned PDE parameters against the ground-truth values used to
generate the synthetic training data in `synthetic_invPDE_4site`.

Reads each result_XX.json (which contains initial params, parameter_history,
and loss_history) and:
  1. Prints per-run final learned values and relative errors.
  2. Prints a summary table (mean, std, mean |rel err|) across runs.
  3. Plots parameter trajectories vs epoch for every run, with ground-truth
     reference lines, plus a loss-history plot.

Run from the repo root:
    python compare_invPDE_synthetic_params.py
"""
#%%
import json
import os
from glob import glob

import matplotlib.pyplot as plt
import numpy as np

# Ground truth used by train_invPDE_synthetic_batch.py to generate the data.
from synthetic_ground_truth import GROUND_TRUTH_4SITE as GROUND_TRUTH
PARAM_NAMES = list(GROUND_TRUTH.keys())

RESULTS_DIR = os.path.join("results", "synthetic_invPDE_4site", "results")
OUT_DIR = os.path.join("results", "synthetic_invPDE_4site", "param_comparison")
os.makedirs(OUT_DIR, exist_ok=True)

#%%
def load_runs():
    paths = sorted(glob(os.path.join(RESULTS_DIR, "result_*.json")))
    if not paths:
        raise FileNotFoundError(f"No result_*.json files found in {RESULTS_DIR}")
    runs = []
    for p in paths:
        with open(p) as f:
            runs.append(json.load(f))
    return runs


def final_params(run):
    last = run["parameter_history"][-1]
    return {k: last[k] for k in PARAM_NAMES}


def print_per_run(runs):
    header = f"{'run':>4} {'seed':>6} {'final_loss':>12}  " + "  ".join(
        f"{n[:14]:>14}" for n in PARAM_NAMES
    )
    print("=" * len(header))
    print("LEARNED VALUES")
    print(header)
    print("-" * len(header))
    for r in runs:
        fp = final_params(r)
        row = f"{r['run_id']:>4d} {r['seed']:>6d} {r['final_loss']:>12.4f}  " + \
              "  ".join(f"{fp[n]:>14.5f}" for n in PARAM_NAMES)
        print(row)
    print()
    print("GROUND TRUTH")
    print(" " * 25 + "  ".join(f"{GROUND_TRUTH[n]:>14.5f}" for n in PARAM_NAMES))
    print()


def print_summary(runs):
    learned = np.array([[final_params(r)[n] for n in PARAM_NAMES] for r in runs])
    truth = np.array([GROUND_TRUTH[n] for n in PARAM_NAMES])
    mean = learned.mean(axis=0)
    std = learned.std(axis=0)
    rel_err = np.abs(learned - truth) / truth
    mean_rel_err = rel_err.mean(axis=0)

    print("=" * 90)
    print(f"SUMMARY across {len(runs)} runs")
    print(f"{'parameter':>30} {'truth':>12} {'mean':>12} {'std':>12} {'mean|relerr|':>14}")
    print("-" * 90)
    for i, n in enumerate(PARAM_NAMES):
        print(f"{n:>30} {truth[i]:>12.5f} {mean[i]:>12.5f} "
              f"{std[i]:>12.5f} {mean_rel_err[i]:>14.4f}")
    print()
    print(f"Overall mean |rel err|: {mean_rel_err.mean():.4f}")


def plot_parameter_trajectories(runs):
    fig, axes = plt.subplots(3, 3, figsize=(15, 11), sharex=True)
    axes = axes.flatten()
    PRETTY_NAMES = {
    "infiltration_rate":             "Infiltration rate",
    "seepage_rate":                  "Seepage rate",
    "plant_uptake_rate":             "Plant uptake rate",
    "mortality_rate":                "Mortality rate",
    "evaporation_rate":              "Evaporation rate",
    "water_use_efficiency":          "Water use efficiency",
    "surface_water_diffusion_coeff": "Surface water diffusion",
    "soil_water_diffusion_coeff":    "Soil water diffusion",
    "biomass_diffusion_coeff":       "Biomass diffusion",
    }   


    for i, name in enumerate(PARAM_NAMES):
        ax = axes[i]
        for r in runs:
            epochs = [e["epoch"] for e in r["parameter_history"]]
            vals = [e[name] for e in r["parameter_history"]]
            ax.plot(epochs, vals, alpha=0.5, linewidth=0.9)
        ax.axhline(GROUND_TRUTH[name], color="k", linestyle="--",
                   linewidth=1.5, label="Ground truth")
        ax.set_title(PRETTY_NAMES[name], fontsize=13)
        ax.set_ylabel("Parameter value", fontsize=12)
        if i >= 6:
            ax.set_xlabel("Epoch")
        ax.legend(fontsize=9, loc="best")
        ax.grid(alpha=0.3)
    # fig.suptitle("Learned parameter trajectories vs epoch (all runs)",
    #              fontsize=13)
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "parameter_trajectories.png")
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")

    # Loss history plot
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    for r in runs:
        ax2.plot(r["loss_history"], alpha=0.5, linewidth=0.9,
                 label=f"run {r['run_id']:02d}")
    ax2.set_yscale("log")
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("loss (log)")
    ax2.set_title("Training loss history (all runs)")
    ax2.grid(alpha=0.3)
    fig2.tight_layout()
    out2 = os.path.join(OUT_DIR, "loss_history.png")
    fig2.savefig(out2, dpi=150)
    print(f"Saved {out2}")

    plt.show()


def main():
    runs = load_runs()
    print(f"Loaded {len(runs)} runs from {RESULTS_DIR}\n")
    print_per_run(runs)
    print_summary(runs)
    plot_parameter_trajectories(runs)

#%%
if __name__ == "__main__":
    main()

# %%
