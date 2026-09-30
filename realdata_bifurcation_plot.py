# -*- coding: utf-8 -*-
"""
Bifurcation diagram: average vegetation density vs. annual precipitation.

For every model in the parameter CSV, simulate NUM_YEARS years at each
precipitation level and record the final mean biomass.  The best model is
drawn as a bold black line; all others are thin grey lines.  Spatial
biomass snapshots from the best model are inset at selected precipitation
values.
"""
#%%
import os
import pickle

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from tqdm import tqdm

from realdata_train_invPDE import invRietkerk, RealDataLoader

# ════════════════════════════════════════════════════════════════════════════
# SETTINGS — change these before running
# ════════════════════════════════════════════════════════════════════════════
PRECIP_MIN       = 255.0    # Lower bound of precipitation range [mm]
PRECIP_MAX       = 355.0    # Upper bound of precipitation range [mm]
PRECIP_STEP      = 3.0      # Resolution along the x-axis [mm]
NUM_YEARS        = 1000      # Simulation length per precipitation value

# Precipitation values at which to show spatial inset images (best model)
INSET_PRECIP_VALUES = [265, 280, 295, 310, 325, 340]
# ════════════════════════════════════════════════════════════════════════════

# Paths & fixed config
PARAM_HISTORY_DIR = "results/real_data/models/parameters"
TEST_METRICS_CSV = "results/real_data/test_results/test_metrics.csv"
PARAM_CSV = "results/parameter_history_analysis/four_site_final_parameter_values.csv"
DATA_DIR = "data"
SAVE_DIR = "results/real_data/bifurcation"
NDVI_TO_BIOMASS_MULTIPLIER = 1500.0
STEPS_PER_WEEK = 4

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


def build_model(param_df: pd.DataFrame, model_id: int, device: torch.device) -> invRietkerk:
    row = param_df.loc[model_id]
    model = invRietkerk(trainable=True).to(device)
    with torch.no_grad():
        for name in PARAM_NAMES:
            getattr(model, name).data.fill_(row[name])
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
    # Initialize a year of zero precipitation
    weekly_rates = np.zeros(52)
    
    # June 1st to Aug 31st roughly corresponds to week 21 to 34 (0-indexed)
    start_week = 21
    end_week = 34
    n_rainy_weeks = end_week - start_week + 1
    
    # Create the sinusoidal shape for the summer weeks
    for i, w in enumerate(range(start_week, end_week + 1)):
        # Sine wave from 0 to pi ensures it starts at 0, peaks in the middle, and ends at 0
        weekly_rates[w] = np.sin(np.pi * i / (n_rainy_weeks - 1))
        
    # The PDE expects daily rates (mm/day), but there are 365/52 days in a week.
    # We need to normalize the curve so the total annual precipitation matches annual_mm.
    days_per_week = 365.0 / 52.0
    shape_sum = np.sum(weekly_rates)
    
    # Calculate the multiplier to scale the sine wave to the correct total volume
    # Formula: sum(weekly_rates * days_per_week) = annual_mm
    multiplier = annual_mm / (shape_sum * days_per_week)
    
    # Apply the multiplier to get the final daily rates per week
    weekly_rates = weekly_rates * multiplier
    
    return weekly_rates.tolist()


def filter_tier1(param_df: pd.DataFrame) -> pd.DataFrame:
    """Drop structural failures: short history or degenerate/NaN final params.

    Mirrors filter_tier1() in compare_invPDE_realdata_params.py.
    Reads pkl history files from PARAM_HISTORY_DIR.
    """
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


# ── Main computation ────────────────────────────────────────────────────────
#%%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

print(f"\nSweeping {len(precip_values)} precipitation values "
      f"({PRECIP_MIN}–{PRECIP_MAX} mm) x {n_models} models x {NUM_YEARS} years ...")

# results[i, j] = final mean biomass for model i, precip j
results = np.zeros((n_models, len(precip_values)))
# Store spatial snapshots for the best model at selected precip values
best_spatial = {}

for i, mid in enumerate(tqdm(model_ids, desc="Models")):
    model = build_model(param_df, mid, device)
    is_best = (mid == best_model_id)
    print(f"now on {mid}")
    for j, p in enumerate(precip_values):
        need_spatial = is_best and (round(p) in INSET_PRECIP_VALUES)
        out = simulate_final_biomass(
            model, initial_biomass, p, NUM_YEARS, device,
            return_spatial=need_spatial,
        )
        if need_spatial:
            results[i, j], spatial = out
            best_spatial[p] = spatial
        else:
            results[i, j] = out

print("Simulation sweep complete.")

# Save numerical results
sweep_df = pd.DataFrame(results, index=model_ids, columns=precip_values)
sweep_df.index.name = "model_id"
sweep_df.to_csv(os.path.join(SAVE_DIR, "bifurcation_data.csv"))

#%%
# ── Plot ────────────────────────────────────────────────────────────────────
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

# ── Inset spatial images ────────────────────────────────────────────────────
# Colourmap for insets (same green-yellow as reference figure)
cmap = plt.cm.YlGn
vmax_inset = NDVI_TO_BIOMASS_MULTIPLIER * 0.35  # adjust to get good contrast

for p_val, spatial in sorted(best_spatial.items()):
    mean_val = spatial.mean()

    # Plot a red marker + vertical line at this precipitation value
    ax.plot(p_val, mean_val, "o", color="red", ms=6, zorder=5)
    ax.axvline(p_val, color="red", lw=0.5, alpha=0.4, zorder=1)

    # Render the spatial image as an inset
    norm_inset = Normalize(vmin=0, vmax=vmax_inset)
    rgba = cmap(norm_inset(spatial))
    thumb = OffsetImage(rgba, zoom=0.55)
    thumb.image.axes = ax

    # Place inset above the data point with some vertical offset
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

# ── Colour bar for the insets ───────────────────────────────────────────────
sm = plt.cm.ScalarMappable(cmap=cmap, norm=Normalize(vmin=0, vmax=vmax_inset))
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, pad=0.02, fraction=0.03)
cbar.set_label("Vegetation density [g/m$^2$]", fontsize=11)

# ── Labels ──────────────────────────────────────────────────────────────────
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

# %%
