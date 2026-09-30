# -*- coding: utf-8 -*-
"""
Bifurcation diagram: average vegetation density vs. annual precipitation.

For every model in the parameter CSV, simulate NUM_YEARS years at each
precipitation level and record the final mean biomass.  The best model is
drawn as a bold black line; all others are thin grey lines.  Spatial
biomass snapshots from the best model are inset at selected precipitation
values.

This version parallelises the simulation sweep across CPU cores using
multiprocessing.
"""
#%%
import os
import pickle
from collections import Counter
from multiprocessing import Pool, cpu_count
from functools import partial

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from tqdm import tqdm

from realdata_train_invPDE import invRietkerk, RealDataLoader, fit_steps_per_week

# ════════════════════════════════════════════════════════════════════════════
# SETTINGS — change these before running
# ════════════════════════════════════════════════════════════════════════════
PRECIP_MIN       = 255.0    # Lower bound of precipitation range [mm]
PRECIP_MAX       = 355.0    # Upper bound of precipitation range [mm]
PRECIP_STEP      = 3.0      # Resolution along the x-axis [mm]
NUM_YEARS        = 1000     # Simulation length per precipitation value

# Precipitation values at which to show spatial inset images (best model)
INSET_PRECIP_VALUES = [265, 280, 295, 310, 325, 340]

# Number of parallel worker processes (None = use all available cores)
N_WORKERS = 48
# ════════════════════════════════════════════════════════════════════════════

# Paths & fixed config
PARAM_HISTORY_DIR = "results/real_data/models/parameters"
TEST_METRICS_CSV = "results/real_data/test_results/test_metrics.csv"
PARAM_CSV = "results/parameter_history_analysis/four_site_final_parameter_values.csv"
DATA_DIR = "data"
SAVE_DIR = "results/real_data/bifurcation"
NDVI_TO_BIOMASS_MULTIPLIER = 1500.0
# Must match the fit: the coefficients belong to the training discretisation.
STEPS_PER_WEEK = fit_steps_per_week()

PARAM_NAMES = [
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

# Tier-1 filter thresholds (same as compare_invPDE_realdata_params.py)
LENGTH_FRAC = 0.9   # drop runs with fewer than 90 % of the max snapshot count
DEGEN_MIN   = 1e-4  # drop runs with any final param <= this (or NaN / inf)


# ── Helpers ─────────────────────────────────────────────────────────────────
def find_best_model(test_csv: str) -> int:
    df = pd.read_csv(test_csv)
    mean_mse = df.groupby("model_id")["mse"].mean()
    best_id = int(mean_mse.idxmin())
    print(f"Best model by mean test MSE: model {best_id}  (MSE = {mean_mse[best_id]:.2f})")
    return best_id


def build_model_from_dict(param_dict: dict, device: torch.device) -> invRietkerk:
    """Build a model from a plain dict of parameter values (picklable)."""
    model = invRietkerk(trainable=True).to(device)
    with torch.no_grad():
        for name in PARAM_NAMES:
            getattr(model, name).data.fill_(param_dict[name])
    model.eval()
    return model


def load_initial_conditions(device: torch.device) -> torch.Tensor:
    loader = RealDataLoader(
        DATA_DIR, selected_sites=["a"], device=device, use_weekly_precip=True
    )
    data = loader.get_training_data(ndvi_to_biomass_multiplier=NDVI_TO_BIOMASS_MULTIPLIER)
    ts = data["location_time_series"]["subsite_a"]
    initial_biomass = ts[0]["biomass"].clone().to(device)
    print(f"Initial conditions from subsite_a, year {ts[0]['year']}")
    print(f"  Shape: {tuple(initial_biomass.shape)}, "
          f"range: {initial_biomass.min().item():.1f} – {initial_biomass.max().item():.1f}")
    return initial_biomass


def make_weekly_precip(annual_mm: float) -> list:
    """
    Create a 52-element list of weekly precipitation rates (mm/day).
    Rainfall is restricted to June, July, and August (approx. weeks 21 to 34),
    distributed as a sinusoidal curve.
    """
    weekly_rates = np.zeros(52)
    start_week = 21
    end_week = 34
    n_rainy_weeks = end_week - start_week + 1

    for i, w in enumerate(range(start_week, end_week + 1)):
        weekly_rates[w] = np.sin(np.pi * i / (n_rainy_weeks - 1))

    days_per_week = 365.0 / 52.0
    shape_sum = np.sum(weekly_rates)
    multiplier = annual_mm / (shape_sum * days_per_week)
    weekly_rates = weekly_rates * multiplier

    return weekly_rates.tolist()


def filter_tier1(param_df: pd.DataFrame) -> pd.DataFrame:
    """Drop structural failures: short history or degenerate/NaN final params."""
    model_ids = list(param_df.index)
    histories = {}
    for mid in model_ids:
        pkl_path = os.path.join(PARAM_HISTORY_DIR, f"model_{mid:02d}_params.pkl")
        if not os.path.exists(pkl_path):
            print(f"  Tier 1: no history file for model {mid}, excluding")
            continue
        with open(pkl_path, "rb") as f:
            histories[mid] = pickle.load(f)

    max_len = max((len(h) for h in histories.values()), default=1)
    min_len = LENGTH_FRAC * max_len

    kept_ids, dropped = [], []
    for mid in model_ids:
        if mid not in histories:
            dropped.append((mid, "missing history file"))
            continue
        history = histories[mid]
        reasons = []

        if len(history) < min_len:
            reasons.append(f"short history ({len(history)} < {min_len:.0f} snapshots)")

        last = history[-1]
        bad = []
        for name in PARAM_NAMES:
            v = last.get(name, float("nan"))
            if not np.isfinite(v):
                bad.append(f"{name}=nan/inf")
            elif v <= DEGEN_MIN:
                bad.append(f"{name}={v:.2e}")
        if bad:
            reasons.append("degenerate final value(s): " + ", ".join(bad))

        if reasons:
            dropped.append((mid, "; ".join(reasons)))
        else:
            kept_ids.append(mid)

    n_short = sum(1 for _, r in dropped if "short history" in r)
    n_degen = sum(1 for _, r in dropped if "degenerate" in r)
    n_missing = sum(1 for _, r in dropped if "missing history" in r)

    print(f"\n{'─' * 56}")
    print(f"Tier-1 filter (history length + degenerate params)")
    if dropped:
        print(f"  Dropped {len(dropped)} model(s):")
        for mid, reason in dropped:
            print(f"    model {mid}: {reason}")
        print(f"  Reasons summary:")
        if n_missing:
            print(f"    missing history file : {n_missing}")
        if n_short:
            print(f"    short history        : {n_short}")
        if n_degen:
            print(f"    degenerate params    : {n_degen}")
    else:
        print("  No models dropped.")
    print(f"  Kept: {len(kept_ids)} model(s)")
    print(f"{'─' * 56}\n")

    return param_df.loc[param_df.index.isin(kept_ids)]


def simulate_final_biomass(
    model: invRietkerk,
    initial_biomass: torch.Tensor,
    annual_precip_mm: float,
    num_years: int,
    device: torch.device,
    return_spatial: bool = False,
):
    """
    Simulate and return the final-year mean biomass.
    If return_spatial=True, also return the 2-D biomass array.
    """
    weekly_precip = make_weekly_precip(annual_precip_mm)
    sw = torch.zeros_like(initial_biomass, device=device)
    gw = torch.zeros_like(initial_biomass, device=device)
    b = initial_biomass.clone()

    with torch.no_grad():
        for _ in range(num_years):
            sw, gw, b = model.simulate_year_weekly(
                sw, gw, b,
                weekly_precipitation=weekly_precip,
                steps_per_week=STEPS_PER_WEEK,
            )

    final_mean = b.mean().item()
    if return_spatial:
        return final_mean, b.squeeze().cpu().numpy()
    return final_mean


# ── Worker function (runs in each subprocess) ──────────────────────────────
def _worker(task):
    """
    Each worker rebuilds the model from a plain dict to avoid pickling
    PyTorch modules across processes.  We also pin PyTorch to a single
    thread so that process-level parallelism doesn't fight with internal
    OpenMP/MKL threading.
    """
    mid, j, precip, param_dict, initial_biomass_np, ib_shape, num_years, need_spatial = task

    # Prevent PyTorch from spawning its own threads inside each worker
    torch.set_num_threads(1)
    device = torch.device("cpu")

    # Rebuild model from the serialisable parameter dict
    model = build_model_from_dict(param_dict, device)

    # Reconstruct initial biomass tensor
    ib = torch.tensor(initial_biomass_np, dtype=torch.float32, device=device).reshape(ib_shape)

    out = simulate_final_biomass(model, ib, precip, num_years, device, return_spatial=need_spatial)

    if need_spatial:
        mean_val, spatial = out
        return (mid, j, mean_val, spatial)
    return (mid, j, out, None)


# ── Main computation ────────────────────────────────────────────────────────
#%%
if __name__ == "__main__":
    device = torch.device("cpu")  # workers always use CPU
    print(f"Device: {device}")
    os.makedirs(SAVE_DIR, exist_ok=True)

    # Load shared resources
    param_df = pd.read_csv(PARAM_CSV, index_col=0, sep=None, engine="python")
    param_df = filter_tier1(param_df)
    best_model_id = find_best_model(TEST_METRICS_CSV)
    initial_biomass = load_initial_conditions(device)

    precip_values = np.arange(PRECIP_MIN, PRECIP_MAX + PRECIP_STEP / 2, PRECIP_STEP)
    model_ids = list(param_df.index)
    n_models = len(model_ids)

    # Determine worker count
    n_workers = N_WORKERS or cpu_count()
    n_tasks = n_models * len(precip_values)
    print(f"\nSweeping {len(precip_values)} precipitation values "
          f"({PRECIP_MIN}–{PRECIP_MAX} mm) x {n_models} models x {NUM_YEARS} years")
    print(f"Total tasks: {n_tasks}, workers: {n_workers}")

    # ── Build picklable task list ───────────────────────────────────────────
    # Pre-extract parameter dicts (plain Python floats) so we never pickle
    # a torch model or a pandas row across process boundaries.
    param_dicts = {}
    for mid in model_ids:
        param_dicts[mid] = {name: float(param_df.loc[mid, name]) for name in PARAM_NAMES}

    ib_np = initial_biomass.cpu().numpy().ravel()  # flat array for cheap transfer
    ib_shape = tuple(initial_biomass.shape)

    tasks = []
    for i, mid in enumerate(model_ids):
        is_best = (mid == best_model_id)
        for j, p in enumerate(precip_values):
            need_spatial = is_best and (round(p) in INSET_PRECIP_VALUES)
            tasks.append((mid, j, float(p), param_dicts[mid], ib_np, ib_shape,
                          NUM_YEARS, need_spatial))

    # ── Run parallel sweep ──────────────────────────────────────────────────
    results = np.zeros((n_models, len(precip_values)))
    best_spatial = {}
    n_precip = len(precip_values)
    model_progress = Counter()  # mid -> number of precip values completed

    with Pool(processes=n_workers) as pool:
        for mid, j, mean_val, spatial in tqdm(
            pool.imap_unordered(_worker, tasks),
            total=n_tasks,
            desc="Simulating",
        ):
            i = model_ids.index(mid)
            results[i, j] = mean_val
            if spatial is not None:
                best_spatial[precip_values[j]] = spatial

            model_progress[mid] += 1
            if model_progress[mid] == n_precip:
                n_done = sum(1 for v in model_progress.values() if v == n_precip)
                tqdm.write(f"  ✓ Model {mid} complete  ({n_done}/{n_models} models)")

    print("Simulation sweep complete.")

    # Save numerical results
    sweep_df = pd.DataFrame(results, index=model_ids, columns=precip_values)
    sweep_df.index.name = "model_id"
    sweep_df.to_csv(os.path.join(SAVE_DIR, "bifurcation_data.csv"))

    #%%
    # ── Plot ────────────────────────────────────────────────────────────────
    best_idx = model_ids.index(best_model_id)

    fig, ax = plt.subplots(figsize=(12, 6))

    # Grey lines for all other models
    for i, mid in enumerate(model_ids):
        if mid == best_model_id:
            continue
        ax.plot(precip_values, results[i], color="silver", lw=0.8, alpha=0.6)

    # Dummy for legend
    ax.plot([], [], color="silver", lw=0.8, alpha=0.6, label="Other models")

    # Best model in bold black
    ax.plot(precip_values, results[best_idx], color="black", lw=2.5, label="Best model")

    # ── Inset spatial images ────────────────────────────────────────────────
    cmap = plt.cm.YlGn
    vmax_inset = NDVI_TO_BIOMASS_MULTIPLIER * 0.35

    for p_val, spatial in sorted(best_spatial.items()):
        mean_val = spatial.mean()

        ax.plot(p_val, mean_val, "o", color="red", ms=6, zorder=5)
        ax.axvline(p_val, color="red", lw=0.5, alpha=0.4, zorder=1)

        norm_inset = Normalize(vmin=0, vmax=vmax_inset)
        rgba = cmap(norm_inset(spatial))
        thumb = OffsetImage(rgba, zoom=0.55)
        thumb.image.axes = ax

        y_offset = 50
        ab = AnnotationBbox(
            thumb,
            (p_val, mean_val),
            xybox=(0, y_offset),
            xycoords="data",
            boxcoords="offset points",
            frameon=True,
            bboxprops=dict(edgecolor="grey", lw=0.8),
            arrowprops=dict(arrowstyle="-", color="red", lw=0.8),
        )
        ax.add_artist(ab)

    # ── Colour bar for the insets ───────────────────────────────────────────
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=Normalize(vmin=0, vmax=vmax_inset))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02, fraction=0.03)
    cbar.set_label("Vegetation density [g/m$^2$]", fontsize=11)

    # ── Labels ──────────────────────────────────────────────────────────────
    ax.set_xlabel("Average annual precipitation [mm]", fontsize=12)
    ax.set_ylabel("Average vegetation density [g/m$^2$]", fontsize=12)
    ax.legend(loc="upper left", fontsize=11)
    ax.set_xlim(PRECIP_MIN, PRECIP_MAX)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    fig.savefig(os.path.join(SAVE_DIR, "bifurcation_diagram.png"), dpi=200,
                bbox_inches="tight")
    fig.savefig(os.path.join(SAVE_DIR, "bifurcation_diagram.pdf"),
                bbox_inches="tight")
    plt.show()
    plt.close(fig)

    print(f"\nPlots saved to {SAVE_DIR}/")
    print("  bifurcation_diagram.png")
    print("  bifurcation_diagram.pdf")
    print("  bifurcation_data.csv")
