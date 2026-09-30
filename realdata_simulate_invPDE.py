# -*- coding: utf-8 -*-
"""
Run forward PDE simulation using the best tested model and subsite_a as
initial conditions.  Change the settings below to vary precipitation level,
number of years, and model selection.
"""
#%%
import os
import json

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from realdata_train_invPDE import invRietkerk, RealDataLoader, fit_steps_per_week
from model_selection import best_model_by_test_mse

# ════════════════════════════════════════════════════════════════════════════
# SETTINGS — change these before running
# ════════════════════════════════════════════════════════════════════════════
ANNUAL_PRECIP_MM = 270.0   # Annual precipitation in mm
NUM_YEARS        = 510       # Number of years to simulate
MODEL_ID         = None     # Set to an int to pick a specific model,
                            # or leave as None to auto-select the best one
# ════════════════════════════════════════════════════════════════════════════

# Paths & fixed config (should not need changing)
TEST_METRICS_CSV = "results/real_data/test_results/test_metrics.csv"
PARAM_CSV = "results/parameter_history_analysis/four_site_final_parameter_values.csv"
DATA_DIR = "data"
SAVE_DIR = "results/real_data/simulation_results"
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


# ── Helpers ─────────────────────────────────────────────────────────────────
def find_best_model(test_csv: str) -> int:
    """Return the model_id with the lowest mean MSE across all test sites."""
    best_id, best_mse = best_model_by_test_mse(pd.read_csv(test_csv))
    print(f"Best model by mean test MSE: model {best_id}  (MSE = {best_mse:.2f})")
    return best_id


def build_model(param_df: pd.DataFrame, model_id: int, device: torch.device) -> invRietkerk:
    """Build an invRietkerk model with parameters from the CSV row."""
    row = param_df.loc[model_id]
    model = invRietkerk(trainable=True).to(device)
    with torch.no_grad():
        for name in PARAM_NAMES:
            getattr(model, name).data.fill_(row[name])
    model.eval()
    return model


def load_initial_conditions(device: torch.device):
    """Load the first observation from subsite_a as initial biomass state."""
    loader = RealDataLoader(
        DATA_DIR, selected_sites=["a"], device=device, use_weekly_precip=True
    )
    data = loader.get_training_data(ndvi_to_biomass_multiplier=NDVI_TO_BIOMASS_MULTIPLIER)
    ts = data["location_time_series"]["subsite_a"]

    initial_biomass = ts[0]["biomass"].clone().to(device)
    first_year = ts[0]["year"]
    print(f"Initial conditions from subsite_a, year {first_year}")
    print(f"  Biomass shape: {tuple(initial_biomass.shape)}")
    print(f"  Biomass range: {initial_biomass.min().item():.1f} – {initial_biomass.max().item():.1f}")
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


def simulate(
    model: invRietkerk,
    initial_biomass: torch.Tensor,
    annual_precip_mm: float,
    num_years: int,
    device: torch.device,
) -> dict:
    """
    Run the PDE forward for *num_years* at a constant annual precipitation.

    Returns dict with arrays for surface_water, soil_water, biomass snapshots
    at the end of each year (plus the initial state at index 0).
    """
    weekly_precip = make_weekly_precip(annual_precip_mm)

    sw = torch.zeros_like(initial_biomass, device=device)
    gw = torch.zeros_like(initial_biomass, device=device)
    b = initial_biomass.clone()

    # Store snapshots: index 0 = initial, then one per year
    snapshots = {
        "biomass": [b.squeeze().cpu().numpy()],
        "surface_water": [sw.squeeze().cpu().numpy()],
        "soil_water": [gw.squeeze().cpu().numpy()],
    }

    print(f"\nSimulating {num_years} years at {annual_precip_mm:.1f} mm/yr ...")
    with torch.no_grad():
        for yr in range(1, num_years + 1):
            sw, gw, b = model.simulate_year_weekly(
                sw, gw, b,
                weekly_precipitation=weekly_precip,
                steps_per_week=STEPS_PER_WEEK,
            )
            snapshots["biomass"].append(b.squeeze().cpu().numpy())
            snapshots["surface_water"].append(sw.squeeze().cpu().numpy())
            snapshots["soil_water"].append(gw.squeeze().cpu().numpy())

            if yr % max(1, num_years // 10) == 0 or yr == num_years:
                print(f"  Year {yr:>4d}  |  biomass mean={b.mean().item():.2f}  "
                      f"min={b.min().item():.2f}  max={b.max().item():.2f}")

    return snapshots


# ── Visualisation ───────────────────────────────────────────────────────────
def plot_results(snapshots: dict, annual_precip_mm: float, num_years: int,
                 model_id: int, save_dir: str):
    """Create and save summary figures."""

    biomass = np.array(snapshots["biomass"])  # (num_years+1, H, W)
    n_frames = biomass.shape[0]

    # ── 1. Time series of spatial statistics ────────────────────────────────
    means = [b.mean() for b in biomass]
    stds = [b.std() for b in biomass]
    mins = [b.min() for b in biomass]
    maxs = [b.max() for b in biomass]
    years = list(range(n_frames))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(years, means, "o-", label="mean")
    ax.fill_between(
        years,
        [m - s for m, s in zip(means, stds)],
        [m + s for m, s in zip(means, stds)],
        alpha=0.25,
        label="mean +/- 1 std",
    )
    ax.plot(years, mins, "v--", ms=4, label="min")
    ax.plot(years, maxs, "^--", ms=4, label="max")
    ax.set_xlabel("Simulation year")
    ax.set_ylabel("Biomass")
    ax.set_title(f"Biomass evolution — model {model_id}, {annual_precip_mm:.0f} mm/yr")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, "biomass_timeseries.png"), dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    print("  Saved biomass_timeseries.png")

    # ── 2. Spatial snapshots at selected time points ────────────────────────
    if n_frames <= 10:
        indices = list(range(n_frames))
    else:
        # Show ~10 evenly spaced frames including first and last
        indices = sorted(set(
            [0]
            + list(np.linspace(0, n_frames - 1, 10, dtype=int))
            + [n_frames - 1]
        ))

    ncols = min(5, len(indices))
    nrows = (len(indices) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.8 * nrows))
    axes = np.atleast_2d(axes)

    vmax = max(biomass[0].max(), biomass[-1].max(), 1.0)
    norm = Normalize(vmin=0, vmax=vmax)

    for ax_idx, frame_idx in enumerate(indices):
        r, c = divmod(ax_idx, ncols)
        ax = axes[r, c]
        im = ax.imshow(biomass[frame_idx], cmap="RdYlGn", norm=norm)
        ax.set_title(f"Year {frame_idx}")
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Hide unused axes
    for ax_idx in range(len(indices), nrows * ncols):
        r, c = divmod(ax_idx, ncols)
        axes[r, c].axis("off")

    fig.suptitle(
        f"Biomass snapshots — model {model_id}, {annual_precip_mm:.0f} mm/yr",
        fontsize=14,
    )
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, "biomass_snapshots.png"), dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    print("  Saved biomass_snapshots.png")

    # ── 3. Surface water & soil water final state ───────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, key, title in zip(
        axes,
        ["surface_water", "soil_water"],
        ["Surface water (final)", "Soil water (final)"],
    ):
        data = snapshots[key][-1]
        im = ax.imshow(data, cmap="Blues")
        ax.set_title(title)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(f"Water states — model {model_id}, year {num_years}", fontsize=14)
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, "water_states_final.png"), dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    print("  Saved water_states_final.png")


# ── Main ────────────────────────────────────────────────────────────────────
#%%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Create output sub-folder for this run
run_tag = f"precip{ANNUAL_PRECIP_MM:.0f}_years{NUM_YEARS}"
if MODEL_ID is not None:
    run_tag += f"_model{MODEL_ID}"
save_dir = os.path.join(SAVE_DIR, run_tag)
os.makedirs(save_dir, exist_ok=True)

# 1. Select model
param_df = pd.read_csv(PARAM_CSV, index_col=0)
if MODEL_ID is not None:
    model_id = MODEL_ID
    print(f"Using specified model: {model_id}")
else:
    model_id = find_best_model(TEST_METRICS_CSV)

model = build_model(param_df, model_id, device)
print(f"\nModel {model_id} parameters:")
for name in PARAM_NAMES:
    print(f"  {name}: {getattr(model, name).item():.6f}")

# 2. Load initial conditions from subsite_a
initial_biomass = load_initial_conditions(device)

# 3. Simulate
snapshots = simulate(model, initial_biomass, ANNUAL_PRECIP_MM, NUM_YEARS, device)

# 4. Save config + numerical results
config = {
    "model_id": int(model_id),
    "annual_precip_mm": ANNUAL_PRECIP_MM,
    "num_years": NUM_YEARS,
    "steps_per_week": STEPS_PER_WEEK,
    "ndvi_to_biomass_multiplier": NDVI_TO_BIOMASS_MULTIPLIER,
    "initial_site": "subsite_a",
    "parameters": {
        name: float(getattr(model, name).item()) for name in PARAM_NAMES
    },
}
with open(os.path.join(save_dir, "simulation_config.json"), "w") as f:
    json.dump(config, f, indent=2)

# Save biomass statistics per year
biomass = np.array(snapshots["biomass"])
stats_df = pd.DataFrame({
    "year": list(range(biomass.shape[0])),
    "biomass_mean": [b.mean() for b in biomass],
    "biomass_std": [b.std() for b in biomass],
    "biomass_min": [b.min() for b in biomass],
    "biomass_max": [b.max() for b in biomass],
})
stats_df.to_csv(os.path.join(save_dir, "biomass_stats.csv"), index=False)

# 5. Plots
plot_results(snapshots, ANNUAL_PRECIP_MM, NUM_YEARS, model_id, save_dir)

print(f"\nAll outputs saved to {save_dir}/")

# %%
