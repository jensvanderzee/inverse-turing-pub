
#%%
import json
import os
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
#%%
from synthetic_ground_truth import (
    GROUND_TRUTH_1SITE, GROUND_TRUTH_4SITE, PARAM_NAMES, ground_truth_for_run,
)

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

# (label, results dir, ground truth the regime's data was generated with).
# The two experiments use different coefficients, so each is scored against its own.
REGIMES = [
    ("1 site",  os.path.join("results", "synthetic_invPDE_1site", "results"), GROUND_TRUTH_1SITE),
    ("4 sites", os.path.join("results", "synthetic_invPDE_4site", "results"), GROUND_TRUTH_4SITE),
]
REGIME_COLORS = {"1 site": "#d62728", "4 sites": "#1f77b4"}

# Base font size — all text elements scale with this value.
FONT_SIZE = 15
plt.rcParams.update({
    "font.size":        FONT_SIZE,
    "axes.titlesize":   FONT_SIZE,
    "axes.labelsize":   FONT_SIZE,
    "xtick.labelsize":  FONT_SIZE,
    "ytick.labelsize":  FONT_SIZE,
    "legend.fontsize":  FONT_SIZE - 1,
    "figure.titlesize": FONT_SIZE + 8,
})

# Y-axis window: each panel spans ±Y_MARGIN_FRAC × truth above and below truth.
# E.g. 0.20 → ±20 % of the ground-truth value.
Y_MARGIN_FRAC = 0.55

OUT_DIR = os.path.join("results", "param_comparison_1site_vs_4site")
os.makedirs(OUT_DIR, exist_ok=True)


def load_final_params(results_dir, default_truth):
    """Final learned values per parameter, and the ground truth the runs share."""
    paths = sorted(glob(os.path.join(results_dir, "result_*.json")))
    if not paths:
        raise FileNotFoundError(f"No result_*.json files in {results_dir}")
    finals = {n: [] for n in PARAM_NAMES}
    truth = None
    for p in paths:
        with open(p) as f:
            run = json.load(f)
        run_truth = ground_truth_for_run(run, default_truth)
        if truth is None:
            truth = run_truth
        elif run_truth != truth:
            raise ValueError(f"Runs in {results_dir} record different ground truths")
        last = run["parameter_history"][-1]
        for n in PARAM_NAMES:
            finals[n].append(last[n])
    return {n: np.array(v) for n, v in finals.items()}, truth


def plot_paired_box_strip(regime_data, regime_truths):
    fig, axes = plt.subplots(3, 3, figsize=(15, 12), sharex=False)
    axes = axes.flatten()

    regime_labels = [r[0] for r in REGIMES]
    positions = np.arange(len(regime_labels))
    rng = np.random.default_rng(0)

    for i, name in enumerate(PARAM_NAMES):
        ax = axes[i]
        data = [regime_data[label][name] for label in regime_labels]

        bp = ax.boxplot(
            data,
            positions=positions,
            widths=0.55,
            patch_artist=True,
            showfliers=False,
            medianprops=dict(color="black", linewidth=1.5),
            whiskerprops=dict(color="gray"),
            capprops=dict(color="gray"),
        )
        for patch, label in zip(bp["boxes"], regime_labels):
            patch.set_facecolor(REGIME_COLORS[label])
            patch.set_alpha(0.35)
            patch.set_edgecolor(REGIME_COLORS[label])

        for j, label in enumerate(regime_labels):
            vals = regime_data[label][name]
            jitter = rng.uniform(-0.12, 0.12, size=len(vals))
            ax.scatter(
                np.full_like(vals, positions[j]) + jitter,
                vals,
                color=REGIME_COLORS[label],
                alpha=0.75,
                s=22,
                edgecolor="white",
                linewidth=0.5,
                zorder=3,
            )

        # Each regime has its own ground truth: draw it across that regime's box only.
        truths = [regime_truths[label][name] for label in regime_labels]
        for j, (label, truth) in enumerate(zip(regime_labels, truths)):
            ax.hlines(truth, positions[j] - 0.4, positions[j] + 0.4,
                      color=REGIME_COLORS[label], linestyle="--", linewidth=1.3,
                      label=f"Truth ({label}) = {truth:g}")

        ax.set_ylim(min(truths) * (1 - Y_MARGIN_FRAC), max(truths) * (1 + Y_MARGIN_FRAC))
        ax.set_xticks(positions)
        ax.set_xticklabels(regime_labels)
        ax.set_title(PRETTY_NAMES[name], fontsize=15)
        ax.set_ylabel("Parameter value")
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=10, loc="best", framealpha=0.85)

    fig.suptitle(
        "",
        fontsize=14,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = os.path.join(OUT_DIR, "final_params_1site_vs_4site.png")
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")
    plt.show()


def main():
    regime_data, regime_truths = {}, {}
    for label, path, default_truth in REGIMES:
        regime_data[label], regime_truths[label] = load_final_params(path, default_truth)
        n_runs = len(next(iter(regime_data[label].values())))
        print(f"Loaded {n_runs} runs from {path} ({label})")
    plot_paired_box_strip(regime_data, regime_truths)


#%%
if __name__ == "__main__":
    main()

# %%
