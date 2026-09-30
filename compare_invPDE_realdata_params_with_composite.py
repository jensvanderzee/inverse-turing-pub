
#%%
import csv
import os
import pickle
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import rasterio

# Parameter order (matches compare_invPDE_synthetic_params.py for consistency).
PARAM_NAMES = [
    "infiltration_rate",
    "seepage_rate",
    "plant_uptake_rate",
    "mortality_rate",
    "evaporation_rate",
    "water_use_efficiency",
    "surface_water_diffusion_coeff",
    "soil_water_diffusion_coeff",
    "biomass_diffusion_coeff",
]

RESULTS_DIR      = os.path.join("results", "real_data", "parameters")
TEST_METRICS_CSV = os.path.join("results", "real_data", "test_results", "test_metrics.csv")
OUT_DIR = os.path.join("results", "real_data", "param_comparison")
DATA_DIR = "data"
os.makedirs(OUT_DIR, exist_ok=True)

# Training subsites used for the real-data models (same as in
# realdata_train_invPDE.py).
TRAINING_SUBSITES = ["b", "i", "c", "e"]
NDVI_TO_BIOMASS_MULTIPLIER = 1500.0

# Tier 1 filter thresholds (applied to raw, un-scaled values).
LENGTH_FRAC = 0.9      # drop runs with fewer than 90% of the max snapshot count
DEGEN_MIN = 1e-4       # drop runs with any final param <= this value (or NaN/inf)

# Spatial-unit conversion: multiply these parameters by 900 for display and
# reporting. Filter thresholds above still operate on the raw values.
DISPLAY_SCALE = {
    "surface_water_diffusion_coeff": 900.0,
    "soil_water_diffusion_coeff":    900.0,
    "biomass_diffusion_coeff":       900.0,
    "infiltration_rate":             900.0,
    "plant_uptake_rate":             900.0,
}


def scale(name, value):
    """Return value in display units (raw * DISPLAY_SCALE[name] if present)."""
    return value * DISPLAY_SCALE.get(name, 1.0)


def compute_mean_training_biomass(data_dir=DATA_DIR,
                                  subsites=TRAINING_SUBSITES,
                                  multiplier=NDVI_TO_BIOMASS_MULTIPLIER):
    """Mean biomass across all NDVI images used to train the real-data models.

    NDVI = (NIR - Red)/(NIR + Red); biomass = clip(max(NDVI, 0) * multiplier,
    0, multiplier) — identical to realdata_train_invPDE._load_satellite_image.
    """
    per_image_means = []
    for site in subsites:
        ndvi_dir = os.path.join(data_dir, f"subsite_{site}", f"subsite_{site}_ndvi")
        if not os.path.isdir(ndvi_dir):
            print(f"Warning: NDVI directory not found for subsite {site}: {ndvi_dir}")
            continue
        for fname in sorted(f for f in os.listdir(ndvi_dir)
                            if f.lower().endswith(".tif")):
            fpath = os.path.join(ndvi_dir, fname)
            with rasterio.open(fpath) as src:
                red = src.read(1).astype(np.float32)
                nir = src.read(2).astype(np.float32)
                denom = nir + red
                ndvi = np.zeros_like(denom)
                valid = denom > 0
                ndvi[valid] = (nir[valid] - red[valid]) / denom[valid]
                biomass = np.clip(np.maximum(ndvi, 0) * multiplier, 0, multiplier)
                per_image_means.append(float(biomass.mean()))
    if not per_image_means:
        raise RuntimeError(
            f"No NDVI images found under {data_dir} for subsites {subsites}"
        )
    B = float(np.mean(per_image_means))
    print(
        f"Mean training biomass over {len(per_image_means)} images "
        f"(subsites {subsites}, multiplier={multiplier}): B = {B:.6f}"
    )
    return B


def compute_composite(fp, B):
    """(r_2 B / (l_1 + r_2 B)) * (r_1 B / (l_2 + r_1 B)) * j, using raw params."""
    l1 = fp["evaporation_rate"]
    l2 = fp["seepage_rate"]
    r1 = fp["plant_uptake_rate"]
    r2 = fp["infiltration_rate"]
    j = fp["water_use_efficiency"]
    return (r2 * B / (l1 + r2 * B)) * (r1 * B / (l2 + r1 * B)) * j


#%%
def load_runs():
    paths = sorted(glob(os.path.join(RESULTS_DIR, "model_*_params.pkl")))
    if not paths:
        raise FileNotFoundError(f"No model_*_params.pkl files found in {RESULTS_DIR}")
    runs = []
    for p in paths:
        stem = os.path.basename(p).replace("model_", "").replace("_params.pkl", "")
        try:
            run_id = int(stem)
        except ValueError:
            continue
        with open(p, "rb") as f:
            history = pickle.load(f)
        if not history:
            continue
        runs.append({"run_id": run_id, "parameter_history": history})
    return runs


def final_params(run):
    last = run["parameter_history"][-1]
    return {k: last[k] for k in PARAM_NAMES if k in last}


def filter_tier1(runs):
    """Tier 1 filter: drop structural failures.

    Returns (kept, dropped) where dropped is a list of
    (run_id, tier, reason) tuples.
    """
    max_len = max(len(r["parameter_history"]) for r in runs)
    min_len = LENGTH_FRAC * max_len

    kept, dropped = [], []
    for r in runs:
        history = r["parameter_history"]
        reasons = []

        if len(history) < min_len:
            reasons.append(
                f"short history ({len(history)} < {min_len:.0f} snapshots)"
            )

        fp = final_params(r)
        bad = []
        for name in PARAM_NAMES:
            v = fp.get(name, np.nan)
            if not np.isfinite(v):
                bad.append(f"{name}=nan/inf")
            elif v <= DEGEN_MIN:
                bad.append(f"{name}={v:.2e}")
        if bad:
            reasons.append("degenerate final value(s): " + ", ".join(bad))

        if reasons:
            dropped.append((r["run_id"], 1, "; ".join(reasons)))
        else:
            kept.append(r)
    return kept, dropped


def print_dropped(dropped):
    if not dropped:
        print("Tier 1: no runs dropped.\n")
        return
    print("=" * 78)
    print(f"Tier 1: dropped {len(dropped)} runs")
    print(f"{'run':>4}  {'tier':>4}  reason")
    print("-" * 78)
    for run_id, tier, reason in dropped:
        print(f"{run_id:>4d}  {tier:>4d}  {reason}")
    print()


def print_per_run(runs, B=None):
    cols = list(PARAM_NAMES)
    if B is not None:
        cols = cols + ["composite"]
    header = f"{'run':>4}  " + "  ".join(f"{n[:14]:>14}" for n in cols)
    print("=" * len(header))
    print("LEARNED FINAL VALUES")
    print(header)
    print("-" * len(header))
    for r in runs:
        fp = final_params(r)
        values = [scale(n, fp.get(n, float("nan"))) for n in PARAM_NAMES]
        if B is not None:
            values.append(compute_composite(fp, B))
        row = f"{r['run_id']:>4d}  " + "  ".join(f"{v:>14.5f}" for v in values)
        print(row)
    print()


def print_summary(runs, B=None):
    cols = list(PARAM_NAMES)
    rows = [[scale(n, final_params(r).get(n, np.nan)) for n in PARAM_NAMES]
            for r in runs]
    if B is not None:
        cols = cols + ["composite"]
        for i, r in enumerate(runs):
            rows[i].append(compute_composite(final_params(r), B))

    learned = np.array(rows)
    mean = np.nanmean(learned, axis=0)
    std = np.nanstd(learned, axis=0)
    cv = np.where(np.abs(mean) > 0, std / np.abs(mean), np.nan)

    print("=" * 78)
    print(f"SUMMARY across {len(runs)} runs")
    print(f"{'parameter':>30} {'mean':>12} {'std':>12} {'CV (std/|mean|)':>18}")
    print("-" * 78)
    for i, n in enumerate(cols):
        print(f"{n:>30} {mean[i]:>12.5f} {std[i]:>12.5f} {cv[i]:>18.4f}")
    print()


# Display order and pretty names for the summary table (matches the
# published table structure: diffusion coeffs, then loss/rate terms, then
# infiltration / uptake / WUE, with the derived composite appended).
TABLE_ORDER = [
    "surface_water_diffusion_coeff",
    "soil_water_diffusion_coeff",
    "biomass_diffusion_coeff",
    "evaporation_rate",
    "seepage_rate",
    "mortality_rate",
    "infiltration_rate",
    "plant_uptake_rate",
    "water_use_efficiency",
]
PARAM_LABELS = {
    "surface_water_diffusion_coeff": ("Surface water diffusion", "d_1"),
    "soil_water_diffusion_coeff":    ("Soil water diffusion",    "d_2"),
    "biomass_diffusion_coeff":       ("Biomass diffusion",       "d_3"),
    "evaporation_rate":              ("Evaporation rate",        r"\ell_1"),
    "seepage_rate":                  ("Seepage rate",            r"\ell_2"),
    "mortality_rate":                ("Mortality rate",          r"\ell_3"),
    "infiltration_rate":             ("Infiltration rate",       "r_2"),
    "plant_uptake_rate":             ("Plant uptake rate",       "r_1"),
    "water_use_efficiency":          ("Water use efficiency",    "j"),
}
COMPOSITE_LABEL = (
    "Composite",
    r"\frac{r_2 B}{\ell_1 + r_2 B}\,\frac{r_1 B}{\ell_2 + r_1 B}\,j",
)


def _fmt(value):
    """Pick decimal places based on magnitude (matches the image style:
    2 dp for |v| >= 10, 4 dp otherwise)."""
    return f"{value:.2f}" if abs(value) >= 10 else f"{value:.4f}"


def print_latex_table(runs, B=None):
    """Print and save a LaTeX tabular summary: Parameter | Mean ± std | CV.

    Values for the base parameters are reported in display units (diffusion
    coeffs, infiltration rate, and plant uptake rate are multiplied by 900
    via DISPLAY_SCALE). The composite row, if B is provided, is computed
    per-run from the raw (unscaled) learned parameters and the training-time
    mean biomass B.
    """
    base_rows = np.array(
        [[scale(n, final_params(r).get(n, np.nan)) for n in TABLE_ORDER]
         for r in runs]
    )
    base_mean = np.nanmean(base_rows, axis=0)
    base_std = np.nanstd(base_rows, axis=0)
    base_cv = np.where(np.abs(base_mean) > 0, base_std / np.abs(base_mean), np.nan)

    if B is not None:
        comp_values = np.array(
            [compute_composite(final_params(r), B) for r in runs]
        )
        comp_mean = float(np.nanmean(comp_values))
        comp_std = float(np.nanstd(comp_values))
        comp_cv = (comp_std / abs(comp_mean)) if abs(comp_mean) > 0 else float("nan")

    lines = [
        r"\begin{tabular}{|l|c|c|}",
        r"\hline",
        r"\textbf{Parameter} & \textbf{Mean} & \textbf{CV} \\",
        r"\hline",
    ]
    for i, name in enumerate(TABLE_ORDER):
        label, sym = PARAM_LABELS[name]
        lines.append(
            f"{label} (${sym}$) & "
            f"${_fmt(base_mean[i])} \\pm {_fmt(base_std[i])}$ & "
            f"{base_cv[i]:.4f} \\\\"
        )
        lines.append(r"\hline")
    if B is not None:
        label, sym = COMPOSITE_LABEL
        lines.append(
            f"{label} (${sym}$) & "
            f"${_fmt(comp_mean)} \\pm {_fmt(comp_std)}$ & "
            f"{comp_cv:.4f} \\\\"
        )
        lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    latex = "\n".join(lines)

    # Plain-text echo of the same table for quick inspection.
    print("=" * 78)
    print(f"SUMMARY TABLE across {len(runs)} runs")
    if B is not None:
        print(f"(composite uses mean training biomass B = {B:.4f})")
    print(f"{'Parameter':<28} {'Mean ± std':>24}  {'CV':>8}")
    print("-" * 78)
    for i, name in enumerate(TABLE_ORDER):
        label, sym = PARAM_LABELS[name]
        cell = f"{_fmt(base_mean[i])} ± {_fmt(base_std[i])}"
        print(f"{label + ' (' + sym + ')':<28} {cell:>24}  {base_cv[i]:>8.4f}")
    if B is not None:
        label, _sym = COMPOSITE_LABEL
        cell = f"{_fmt(comp_mean)} ± {_fmt(comp_std)}"
        print(f"{label:<28} {cell:>24}  {comp_cv:>8.4f}")
    print()

    out = os.path.join(OUT_DIR, "summary_table_with_composite.tex")
    with open(out, "w") as f:
        f.write(latex + "\n")
    print(f"Saved {out}\n")


def load_test_mse():
    """Return {model_id: mean_mse} averaged across test sites from test_metrics.csv."""
    mse_map = {}
    try:
        site_totals, site_counts = {}, {}
        with open(TEST_METRICS_CSV, newline="") as f:
            for row in csv.DictReader(f):
                mid = int(row["model_id"])
                mse_str = row.get("mse", "").strip()
                if mse_str:
                    site_totals[mid] = site_totals.get(mid, 0.0) + float(mse_str)
                    site_counts[mid]  = site_counts.get(mid, 0) + 1
        for mid, total in site_totals.items():
            if site_counts[mid] > 0:
                mse_map[mid] = total / site_counts[mid]
    except FileNotFoundError:
        print(f"Warning: {TEST_METRICS_CSV} not found; test MSE column will be NaN.")
    return mse_map


def _fmt_allmodels(value):
    """2 dp for |v| >= 10, 3 dp otherwise (matches the per-model table style)."""
    return f"{value:.2f}" if abs(value) >= 10 else f"{value:.3f}"


def _cell_param(raw, name):
    """Format one parameter cell: sentinel for degenerate raw values, else scaled."""
    if not np.isfinite(raw) or raw <= DEGEN_MIN:
        return r"$<$ 0.0001"
    return _fmt_allmodels(scale(name, raw))


def print_latex_all_models_table(runs, B=None):
    """Print and save a per-model LaTeX table with all runs, sorted by test MSE.

    Columns follow the published table structure (TABLE_ORDER) plus an extra
    Rain use efficiency (phi) column computed as the composite:
        phi = (r_2 B / (l_1 + r_2 B)) * (r_1 B / (l_2 + r_1 B)) * j
    using mean training biomass B.  The phi column is omitted when B is None.
    """
    mse_map = load_test_mse()

    rows = []
    for r in runs:
        fp  = final_params(r)
        mse = mse_map.get(r["run_id"], float("nan"))
        comp = compute_composite(fp, B) if B is not None else float("nan")
        rows.append((mse, r["run_id"], fp, comp))
    rows.sort(key=lambda x: x[0])

    has_phi = B is not None
    ncols   = 12 if has_phi else 11
    col_spec = "|".join(["c"] * ncols)

    def row_(*cells):
        return "    " + " & ".join(cells) + r" \\"

    lines = [
        r"\begin{table}[htbp]",
        (r"    \caption{Learned parameters of 30 training runs using satellite data, "
         r"including their MSE on a test set of three locations. "
         r"Parameter values of $<$ 0.0001 indicate failed training runs "
         r"that got stuck in a bad local minimum.}"),
        r"    \label{tab:all_models_realdata}",
        rf"    \begin{{tabular}}{{{col_spec}}}",
        r"    \toprule",
    ]

    # Three-line header
    h1 = [r"\textbf{Model}", r"\textbf{Test}", r"\textbf{Surface}", r"\textbf{Soil}",
          r"\textbf{Biomass}", r"\textbf{Evapor-}", r"\textbf{Seepage}",
          r"\textbf{Mortality}", r"\textbf{Infiltr-}", r"\textbf{Plant}", r"\textbf{Water}"]
    h2 = [r"\textbf{ID}", r"\textbf{MSE}", r"\textbf{water}", r"\textbf{water}",
          r"\textbf{diff.}", r"\textbf{ation}", r"\textbf{rate}", r"\textbf{rate}",
          r"\textbf{ation}", r"\textbf{uptake}", r"\textbf{use}"]
    h3 = [r"\textbf{}", r"\textbf{}",
          r"\textbf{diff. ($d_1$)}", r"\textbf{diff. ($d_2$)}", r"\textbf{($d_3$)}",
          r"\textbf{rate ($\ell_1$)}", r"\textbf{($\ell_2$)}", r"\textbf{($\ell_3$)}",
          r"\textbf{rate ($r_2$)}", r"\textbf{rate ($r_1$)}", r"\textbf{efficiency ($j$)}"]
    if has_phi:
        h1.append(r"\textbf{Rain}")
        h2.append(r"\textbf{use}")
        h3.append(r"\textbf{efficiency ($\phi$)}")
    lines += [row_(*h1), row_(*h2), row_(*h3)]

    # Units row
    units = [r"", r"",
             r"$m^{2}/yr$", r"$m^{2}/yr$", r"$m^{2}/yr$",
             r"$1/yr$", r"$1/yr$", r"$1/yr$",
             r"$m^{2}/(kg\, yr)$", r"$m^{2}/(kg\, yr)$", r"$1$"]
    if has_phi:
        units.append(r"$1$")
    lines += [r"    \hline", row_(*units), r"    \hline", r"    ", r"    \midrule"]

    # Data rows
    for mse, run_id, fp, comp in rows:
        cells = [str(run_id), f"{mse:.2f}"]
        for name in TABLE_ORDER:
            cells.append(_cell_param(fp.get(name, float("nan")), name))
        if has_phi:
            phi_degen = any(
                fp.get(n, float("nan")) <= DEGEN_MIN
                for n in ("infiltration_rate", "plant_uptake_rate", "water_use_efficiency")
            )
            if phi_degen or not np.isfinite(comp):
                cells.append(r"$<$ 0.0001")
            else:
                cells.append(_fmt_allmodels(comp))
        lines.append(row_(*cells))

    lines += [r"    \bottomrule", r"    \end{tabular}", r"\end{table}"]

    latex = "\n".join(lines)
    out = os.path.join(OUT_DIR, "all_models_table.tex")
    with open(out, "w") as f:
        f.write(latex + "\n")
    print(f"Saved {out}\n")


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
            epochs = [e["epoch"] for e in r["parameter_history"] if name in e]
            vals = [scale(name, e[name]) for e in r["parameter_history"]
                    if name in e]
            if epochs:
                ax.plot(epochs, vals, alpha=0.5, linewidth=0.9)
        ax.set_title(PRETTY_NAMES[name], fontsize=13)
        ax.set_ylabel("Parameter value", fontsize=12)
        if i >= 6:
            ax.set_xlabel("Epoch")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "parameter_trajectories.png")
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")

    plt.show()


def main():
    runs = load_runs()
    print(f"Loaded {len(runs)} runs from {RESULTS_DIR}\n")

    try:
        B = compute_mean_training_biomass()
    except Exception as e:
        print(f"Could not compute mean training biomass ({e}); "
              f"composite row will be skipped.")
        B = None

    kept, dropped = filter_tier1(runs)
    print_dropped(dropped)

    print_per_run(kept, B=B)
    print_summary(kept, B=B)
    print_latex_table(kept, B=B)
    print_latex_all_models_table(kept, B=B)
    plot_parameter_trajectories(kept)

#%%
if __name__ == "__main__":
    main()

# %%
