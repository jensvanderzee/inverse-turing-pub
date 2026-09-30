
#%%
import json
import os
from glob import glob

import matplotlib.pyplot as plt
import numpy as np

#%%
GROUND_TRUTH = {
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
PARAM_NAMES = list(GROUND_TRUTH.keys())

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

REGIMES = [
    ("1 site",  os.path.join("results", "synthetic_invPDE_1site", "results")),
    ("4 sites", os.path.join("results", "synthetic_invPDE_4site", "results")),
]
REGIME_COLORS = {"1 site": "#d62728", "4 sites": "#1f77b4"}

FONT_SIZE = 13
plt.rcParams.update({
    "font.size":        FONT_SIZE,
    "axes.titlesize":   FONT_SIZE,
    "axes.labelsize":   FONT_SIZE,
    "xtick.labelsize":  FONT_SIZE,
    "ytick.labelsize":  FONT_SIZE,
    "legend.fontsize":  FONT_SIZE - 1,
    "figure.titlesize": FONT_SIZE + 2,
})

OUT_DIR = os.path.join("results", "param_comparison_1site_vs_4site")
os.makedirs(OUT_DIR, exist_ok=True)


def load_rel_errors(results_dir):
    """Returns array of shape (n_runs, n_params) of |learned - truth| / truth."""
    paths = sorted(glob(os.path.join(results_dir, "result_*.json")))
    if not paths:
        raise FileNotFoundError(f"No result_*.json files in {results_dir}")
    rows = []
    for p in paths:
        with open(p) as f:
            run = json.load(f)
        last = run["parameter_history"][-1]
        row = [abs(last[n] - GROUND_TRUTH[n]) / GROUND_TRUTH[n] for n in PARAM_NAMES]
        rows.append(row)
    return np.array(rows)


def plot_relerr(regime_errors):
    n_params = len(PARAM_NAMES)
    regime_labels = [r[0] for r in REGIMES]
    n_regimes = len(regime_labels)
    rng = np.random.default_rng(0)

    # Each parameter gets a group; within each group the two regimes are offset.
    group_width = 1.0
    regime_offset = 0.22
    group_centers = np.arange(n_params, dtype=float)

    fig, ax = plt.subplots(figsize=(14, 5))

    handles = []
    for ri, label in enumerate(regime_labels):
        color = REGIME_COLORS[label]
        sign = -1 if ri == 0 else 1
        pos = group_centers + sign * regime_offset

        data = [regime_errors[label][:, pi] for pi in range(n_params)]

        bp = ax.boxplot(
            data,
            positions=pos,
            widths=0.3,
            patch_artist=True,
            showfliers=False,
            medianprops=dict(color="black", linewidth=1.5),
            whiskerprops=dict(color=color, linewidth=1.0),
            capprops=dict(color=color, linewidth=1.0),
        )
        for patch in bp["boxes"]:
            patch.set_facecolor(color)
            patch.set_alpha(0.35)
            patch.set_edgecolor(color)

        for pi in range(n_params):
            vals = regime_errors[label][:, pi]
            jitter = rng.uniform(-0.07, 0.07, size=len(vals))
            ax.scatter(
                np.full_like(vals, pos[pi]) + jitter,
                vals,
                color=color,
                alpha=0.65,
                s=18,
                edgecolor="white",
                linewidth=0.4,
                zorder=3,
            )

        handles.append(
            plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=color,
                       markersize=8, label=label)
        )

    # Mean |rel err| per regime as a horizontal summary line
    for ri, label in enumerate(regime_labels):
        mean_overall = regime_errors[label].mean()
        ax.axhline(mean_overall, color=REGIME_COLORS[label], linestyle=":",
                   linewidth=1.4, alpha=0.8,
                   label=f"{label} mean = {mean_overall:.3f}")

    ax.set_xticks(group_centers)
    ax.set_xticklabels([PRETTY_NAMES[n] for n in PARAM_NAMES], rotation=25, ha="right")
    ax.set_ylabel("Absolute relative error  |learned − truth| / truth")
    ax.set_xlim(-0.6, n_params - 0.4)
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(handles=handles + ax.get_lines()[-2:], loc="upper right", framealpha=0.85)

    fig.tight_layout()
    out = os.path.join(OUT_DIR, "relerr_1site_vs_4site.png")
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")
    plt.show()


def print_table(regime_errors):
    regime_labels = [r[0] for r in REGIMES]
    col_w = 22

    header = f"{'Parameter':<30}" + "".join(
        f"{'mean |rel err|':>{col_w}}{'std':>{col_w // 2}}"
        for _ in regime_labels
    )
    subheader = f"{'':30}" + "".join(
        f"{label:>{col_w + col_w // 2}}" for label in regime_labels
    )
    sep = "-" * len(header)

    print(sep)
    print(subheader)
    print(header)
    print(sep)
    for pi, name in enumerate(PARAM_NAMES):
        row = f"{PRETTY_NAMES[name]:<30}"
        for label in regime_labels:
            col = regime_errors[label][:, pi]
            row += f"{col.mean():>{col_w}.4f}{col.std():>{col_w // 2}.4f}"
        print(row)
    print(sep)
    row = f"{'Overall mean':<30}"
    for label in regime_labels:
        overall = regime_errors[label].mean()
        row += f"{overall:>{col_w}.4f}{'':>{col_w // 2}}"
    print(row)
    print(sep)

    # Also save as CSV
    import csv
    out_csv = os.path.join(OUT_DIR, "relerr_summary.csv")
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["parameter"] + [
            f"{label}_mean_relerr" for label in regime_labels
        ] + [
            f"{label}_std_relerr" for label in regime_labels
        ])
        for pi, name in enumerate(PARAM_NAMES):
            writer.writerow(
                [PRETTY_NAMES[name]]
                + [f"{regime_errors[label][:, pi].mean():.6f}" for label in regime_labels]
                + [f"{regime_errors[label][:, pi].std():.6f}"  for label in regime_labels]
            )
        writer.writerow(
            ["Overall mean"]
            + [f"{regime_errors[label].mean():.6f}" for label in regime_labels]
            + [""] * len(regime_labels)
        )
    print(f"Saved {out_csv}")


def main():
    regime_errors = {}
    for label, path in REGIMES:
        regime_errors[label] = load_rel_errors(path)
        print(f"Loaded {regime_errors[label].shape[0]} runs from {path} ({label})")
    print_table(regime_errors)
    plot_relerr(regime_errors)


#%%
if __name__ == "__main__":
    main()

# %%
