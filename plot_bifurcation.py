#%%
import os
import pickle

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

from realdata_train_invPDE import invRietkerk, RealDataLoader, fit_steps_per_week
from model_selection import best_model_by_test_mse

#%%
# ── Paths ────────────────────────────────────────────────────────────────────
DATA_CSV         = "results/real_data/bifurcation/bifurcation_data.csv"
TEST_METRICS_CSV = "results/real_data/test_results/test_metrics.csv"
PARAM_CSV        = "results/parameter_history_analysis/four_site_final_parameter_values.csv"
PARAM_HISTORY_DIR = "results/real_data/models/parameters"
DATA_DIR         = "data"
SAVE_DIR         = "results/real_data/bifurcation"
SNAPSHOTS_PKL    = os.path.join(SAVE_DIR, "spatial_snapshots.pkl")

# Precipitation values (mm) at which to capture a spatial snapshot.
# These must fall on values present in the CSV columns.
INSET_PRECIP_VALUES = [267, 285, 294, 303, 318]

NUM_YEARS            = 1000
# Must match the fit: the coefficients belong to the training discretisation.
STEPS_PER_WEEK       = fit_steps_per_week()
NDVI_TO_BIOMASS_MULTIPLIER = 1500.0

# Tier-1 filter thresholds (must match training script)
LENGTH_FRAC = 0.9
DEGEN_MIN   = 1e-4

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

#%%
# ── Load bifurcation data ────────────────────────────────────────────────────
sweep_df = pd.read_csv(DATA_CSV, index_col=0)
precip_values = sweep_df.columns.astype(float).values
model_ids     = list(sweep_df.index)
results       = sweep_df.values  # (n_models, n_precip)

# Identify best model by mean test MSE
test_df  = pd.read_csv(TEST_METRICS_CSV)
best_id, best_mse = best_model_by_test_mse(test_df)
print(f"Best model: {best_id}  (mean MSE = {best_mse:.4f})")

best_idx = model_ids.index(best_id)

#%%
# ── Helpers ──────────────────────────────────────────────────────────────────
def filter_tier1(param_df: pd.DataFrame) -> pd.DataFrame:
    model_ids_all = list(param_df.index)
    histories = {}
    for mid in model_ids_all:
        pkl_path = os.path.join(PARAM_HISTORY_DIR, f"model_{mid:02d}_params.pkl")
        if os.path.exists(pkl_path):
            with open(pkl_path, "rb") as f:
                histories[mid] = pickle.load(f)

    max_len = max((len(h) for h in histories.values()), default=1)
    min_len = LENGTH_FRAC * max_len
    kept_ids = []
    for mid in model_ids_all:
        if mid not in histories or len(histories[mid]) < min_len:
            continue
        last = histories[mid][-1]
        if any(not np.isfinite(last.get(n, float("nan"))) or last.get(n, 0) <= DEGEN_MIN
               for n in PARAM_NAMES):
            continue
        kept_ids.append(mid)
    return param_df.loc[param_df.index.isin(kept_ids)]


def build_model(param_df: pd.DataFrame, model_id: int, device: torch.device) -> invRietkerk:
    row = param_df.loc[model_id]
    model = invRietkerk(trainable=True).to(device)
    with torch.no_grad():
        for name in PARAM_NAMES:
            getattr(model, name).data.fill_(row[name])
    model.eval()
    return model


def load_initial_conditions(device: torch.device) -> torch.Tensor:
    loader = RealDataLoader(DATA_DIR, selected_sites=["a"], device=device, use_weekly_precip=True)
    data = loader.get_training_data(ndvi_to_biomass_multiplier=NDVI_TO_BIOMASS_MULTIPLIER)
    ts = data["location_time_series"]["subsite_a"]
    return ts[0]["biomass"].clone().to(device)


def make_weekly_precip(annual_mm: float) -> list:
    rates = np.zeros(52)
    start_week, end_week = 21, 34
    n_rainy = end_week - start_week + 1
    for i, w in enumerate(range(start_week, end_week + 1)):
        rates[w] = np.sin(np.pi * i / (n_rainy - 1))
    days_per_week = 365.0 / 52.0
    rates *= annual_mm / (rates.sum() * days_per_week)
    return rates.tolist()


def run_snapshot(model: invRietkerk, initial_biomass: torch.Tensor,
                 annual_mm: float, num_years: int, device: torch.device) -> np.ndarray:
    weekly_precip = make_weekly_precip(annual_mm)
    sw = torch.zeros_like(initial_biomass)
    gw = torch.zeros_like(initial_biomass)
    b  = initial_biomass.clone()
    with torch.no_grad():
        for _ in range(num_years):
            sw, gw, b = model.simulate_year_weekly(
                sw, gw, b,
                weekly_precipitation=weekly_precip,
                steps_per_week=STEPS_PER_WEEK,
            )
    return b.squeeze().cpu().numpy()


#%%
# ── Spatial snapshots ────────────────────────────────────────────────────────
# Snapshots are cached so you don't re-run the simulation every time you tweak the plot.
if os.path.exists(SNAPSHOTS_PKL):
    with open(SNAPSHOTS_PKL, "rb") as f:
        spatial_snapshots = pickle.load(f)
    print(f"Loaded cached snapshots for precip values: {sorted(spatial_snapshots.keys())}")
    missing = [p for p in INSET_PRECIP_VALUES if p not in spatial_snapshots]
else:
    spatial_snapshots = {}
    missing = list(INSET_PRECIP_VALUES)

if missing:
    print(f"Running {NUM_YEARS}-year simulations for precip values: {missing}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    param_df = pd.read_csv(PARAM_CSV, index_col=0, sep=None, engine="python")
    param_df = filter_tier1(param_df)
    model    = build_model(param_df, best_id, device)
    initial_biomass = load_initial_conditions(device)

    for p in missing:
        print(f"  Simulating {p} mm ...", end=" ", flush=True)
        spatial_snapshots[p] = run_snapshot(model, initial_biomass, float(p), NUM_YEARS, device)
        print(f"done  (mean biomass = {spatial_snapshots[p].mean():.1f} g/m²)")

    os.makedirs(SAVE_DIR, exist_ok=True)
    with open(SNAPSHOTS_PKL, "wb") as f:
        pickle.dump(spatial_snapshots, f)
    print(f"Snapshots saved to {SNAPSHOTS_PKL}")

#%%
# ── Plot ─────────────────────────────────────────────────────────────────────
FONT_SIZE = 16  # change this to scale all text at once
plt.rcParams.update({"font.size": FONT_SIZE})

cmap   = plt.cm.YlGn
vmax_inset = NDVI_TO_BIOMASS_MULTIPLIER * 0.3

fig, ax = plt.subplots(figsize=(12, 8))

for i, mid in enumerate(model_ids):
    if mid == best_id:
        continue
    ax.plot(precip_values, results[i], color="silver", lw=0.8, alpha=0.6)

ax.plot([], [], color="silver", lw=0.8, alpha=0.6, label="Other models")
ax.plot(precip_values, results[best_idx], color="black", lw=2.5, label="Best model")

# ── Inset spatial snapshots ──────────────────────────────────────────────────
norm_inset = Normalize(vmin=0, vmax=vmax_inset)

inset_precip_sorted = sorted(p for p in spatial_snapshots if p in INSET_PRECIP_VALUES)

for rank, p_val in enumerate(inset_precip_sorted):
    spatial = spatial_snapshots[p_val]

    col_idx  = np.argmin(np.abs(precip_values - float(p_val)))
    mean_val = results[best_idx, col_idx]

    ax.plot(precip_values[col_idx], mean_val, "o", color="red", ms=6, zorder=5)

    rgba  = cmap(norm_inset(spatial))
    thumb = OffsetImage(rgba, zoom=0.55)
    thumb.image.axes = ax

    is_last  = (rank == len(inset_precip_sorted) - 1)
    y_offset = 81 if is_last else 145

    ab = AnnotationBbox(
        thumb,
        (precip_values[col_idx], mean_val),
        xybox=(0, y_offset),
        xycoords="data",
        boxcoords="offset points",
        frameon=True,
        bboxprops=dict(edgecolor="black", lw=0.8),
        arrowprops=dict(arrowstyle="-", color="red", lw=0.8),
    )
    ax.add_artist(ab)

# ── Colourbar for insets ─────────────────────────────────────────────────────
sm = plt.cm.ScalarMappable(cmap=cmap, norm=Normalize(vmin=0, vmax=vmax_inset))
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, pad=0.02, fraction=0.03)
cbar.set_label("Vegetation density [g/m$^2$]", fontsize=15)

ax.set_xlabel("Average annual precipitation [mm]", fontsize=18)
ax.set_ylabel("Average vegetation density [g/m$^2$]", fontsize=18)
ax.legend(loc="upper left", fontsize=15)
ax.set_xlim(precip_values[0], 335)
ax.set_ylim(bottom=0)

plt.tight_layout()
os.makedirs(SAVE_DIR, exist_ok=True)
fig.savefig(os.path.join(SAVE_DIR, "bifurcation_diagram.png"), dpi=200, bbox_inches="tight")
fig.savefig(os.path.join(SAVE_DIR, "bifurcation_diagram.pdf"), bbox_inches="tight")

plt.show()
plt.close(fig)

print(f"Saved to {SAVE_DIR}/bifurcation_diagram.png/.pdf")

# %%
