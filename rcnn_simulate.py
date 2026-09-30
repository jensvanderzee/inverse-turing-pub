# -*- coding: utf-8 -*-
"""
Run forward simulation using the best retrained RCNN model.
Uses synthetic PDE equilibrium state as initial conditions (matching training).
Precipitation level and number of years can be changed via the settings below.

Model architecture (RCNNBaseline + ConvRNNCell) and data generation
(EcologicalParameters + invRietkerk + SyntheticDataGenerator) are copied
from train_rcnn_batch.py to avoid import side-effects.
"""
#%%
import os
import json
import math
import random
from typing import List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize


# ── Reproducibility (from train_rcnn_batch.py) ────────────────────────────
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ── Ground-truth PDE model (from train_rcnn_batch.py) ─────────────────────
class EcologicalParameters:
    SURFACE_WATER_DIFFUSION = 8.0
    SOIL_WATER_DIFFUSION = 1.0
    BIOMASS_DIFFUSION = 0.05
    EVAPORATION_RATE = 0.3
    SEEPAGE_RATE = 0.4
    MORTALITY_RATE = 0.3
    INFILTRATION_RATE = 0.1
    PLANT_UPTAKE_RATE = 0.15
    BASE_PRECIPITATION = 0.675
    WATER_USE_EFFICIENCY = 0.35

    TIME_STEP = 0.3
    NOISE_LEVEL = 0.05
    STEPS_PER_WEEK = 1
    NUM_WEEKS = 52


class invRietkerk(nn.Module):
    """Ground-truth (non-trainable) PDE simulator used only for data generation."""

    def __init__(self, trainable: bool = False):
        super().__init__()
        self.base_time_step = EcologicalParameters.TIME_STEP
        self.trainable = trainable
        self._setup_spatial_operators()
        self._initialize_parameters()

    def _setup_spatial_operators(self):
        laplacian_kernel = torch.tensor(
            [[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32
        )
        for name in ["surface_water", "soil_water", "biomass"]:
            conv = nn.Conv2d(1, 1, kernel_size=3, padding=1, padding_mode="replicate", bias=False)
            conv.weight = nn.Parameter(laplacian_kernel[None, None, :], requires_grad=False)
            setattr(self, f"{name}_diffusion_op", conv)

    def _initialize_parameters(self):
        self.infiltration_rate = EcologicalParameters.INFILTRATION_RATE
        self.evaporation_rate = EcologicalParameters.EVAPORATION_RATE
        self.seepage_rate = EcologicalParameters.SEEPAGE_RATE
        self.plant_uptake_rate = EcologicalParameters.PLANT_UPTAKE_RATE
        self.mortality_rate = EcologicalParameters.MORTALITY_RATE
        self.water_use_efficiency = EcologicalParameters.WATER_USE_EFFICIENCY
        self.surface_water_diffusion_coeff = EcologicalParameters.SURFACE_WATER_DIFFUSION
        self.soil_water_diffusion_coeff = EcologicalParameters.SOIL_WATER_DIFFUSION
        self.biomass_diffusion_coeff = EcologicalParameters.BIOMASS_DIFFUSION

    def forward(self, surface_water, soil_water, biomass, precipitation_rate=0.0, time_step=None):
        if time_step is None:
            time_step = self.base_time_step
        surface_water = surface_water + (
            self.surface_water_diffusion_coeff * self.surface_water_diffusion_op(surface_water)
            - self.evaporation_rate * surface_water
            + precipitation_rate
            - self.infiltration_rate * surface_water * biomass
        ) * time_step
        soil_water = soil_water + (
            self.soil_water_diffusion_coeff * self.soil_water_diffusion_op(soil_water)
            - self.seepage_rate * soil_water
            + self.infiltration_rate * surface_water * biomass
            - self.plant_uptake_rate * soil_water * biomass
        ) * time_step
        biomass = biomass + (
            self.biomass_diffusion_coeff * self.biomass_diffusion_op(biomass)
            - self.mortality_rate * biomass
            + self.water_use_efficiency * self.plant_uptake_rate * soil_water * biomass
        ) * time_step
        return surface_water, soil_water, biomass

    def simulate_year_weekly(self, surface_water, soil_water, biomass, weekly_precipitation, steps_per_week=3):
        cur_sw, cur_soilw, cur_bio = surface_water.clone(), soil_water.clone(), biomass.clone()
        num_weeks = len(weekly_precipitation)
        total_steps = num_weeks * steps_per_week
        dt = 1.0 / total_steps
        for week in range(num_weeks):
            real_weekly_volume = weekly_precipitation[week] * 7.0
            volume_per_step = real_weekly_volume / steps_per_week
            adjusted_precip_rate = volume_per_step / dt
            for _ in range(steps_per_week):
                cur_sw, cur_soilw, cur_bio = self.forward(
                    cur_sw, cur_soilw, cur_bio,
                    precipitation_rate=adjusted_precip_rate, time_step=dt,
                )
        return cur_sw, cur_soilw, cur_bio


class SyntheticDataGenerator:
    def __init__(self, grid_size=(150, 150), device=None):
        self.height, self.width = grid_size
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ground_truth_model = invRietkerk(trainable=False).to(self.device)

    def generate_equilibrium_state(self, equilibrium_precipitation=10.5, equilibrium_years=100, steps_per_week=3):
        surface_water = torch.rand(1, 1, self.height, self.width, device=self.device) * 10
        soil_water = torch.rand(1, 1, self.height, self.width, device=self.device) * 10
        biomass = torch.rand(1, 1, self.height, self.width, device=self.device) * 10
        uniform_weekly = generate_weekly_precipitation(
            equilibrium_precipitation, peak_week=26.0, amplitude_fraction=0.0,
        )
        for _ in range(equilibrium_years):
            surface_water, soil_water, biomass = self.ground_truth_model.simulate_year_weekly(
                surface_water, soil_water, biomass,
                weekly_precipitation=uniform_weekly, steps_per_week=steps_per_week,
            )
        return biomass.clone()


# ── RCNN model definition (from train_rcnn_batch.py) ──────────────────────
class ConvRNNCell(nn.Module):
    def __init__(self, input_channels: int, hidden_channels: int, kernel_size: int = 3):
        super().__init__()
        self.hidden_channels = hidden_channels
        pad = kernel_size // 2
        self.conv = nn.Conv2d(
            input_channels + hidden_channels, hidden_channels,
            kernel_size=kernel_size, padding=pad, padding_mode="replicate",
        )

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.conv(torch.cat([x, h], dim=1)))


class RCNNBaseline(nn.Module):
    def __init__(self, hidden_channels: int = 32, num_layers: int = 1, kernel_size: int = 3):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.encoder = nn.Conv2d(2, hidden_channels, kernel_size=1)
        self.rnn_cells = nn.ModuleList()
        for _ in range(num_layers):
            self.rnn_cells.append(ConvRNNCell(hidden_channels, hidden_channels, kernel_size))
        self.decoder = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 1, kernel_size=1),
        )

    def _init_hidden(self, batch_size, height, width, device):
        return [
            torch.zeros(batch_size, self.hidden_channels, height, width, device=device)
            for _ in range(self.num_layers)
        ]

    def forward_one_week(self, biomass, precip_map, hidden_states):
        x = torch.cat([biomass, precip_map], dim=1)
        x = self.encoder(x)
        new_hidden = []
        for layer_idx, cell in enumerate(self.rnn_cells):
            h = cell(x, hidden_states[layer_idx])
            new_hidden.append(h)
            x = h
        delta = self.decoder(x)
        next_biomass = biomass + delta
        return next_biomass, new_hidden

    def simulate_year_weekly(self, biomass, weekly_precipitation, hidden_states=None):
        B, _, H, W = biomass.shape
        device = biomass.device
        if hidden_states is None:
            hidden_states = self._init_hidden(B, H, W, device)
        current_biomass = biomass
        for week_idx in range(len(weekly_precipitation)):
            precip_val = weekly_precipitation[week_idx] * 7.0
            precip_map = torch.full((B, 1, H, W), precip_val, device=device)
            current_biomass, hidden_states = self.forward_one_week(
                current_biomass, precip_map, hidden_states,
            )
        return current_biomass, hidden_states


# ── Precipitation helper (from train_rcnn_batch.py) ─────────────────────────
def generate_weekly_precipitation(
    annual_total: float,
    peak_week: float = 26.0,
    amplitude_fraction: float = 0.7,
    num_weeks: int = 52,
) -> List[float]:
    mean_rate = annual_total / 365.0
    raw = [
        mean_rate
        * (1.0 + amplitude_fraction * math.cos(2.0 * math.pi * (w - peak_week) / num_weeks))
        for w in range(num_weeks)
    ]
    raw_total = sum(r * 7.0 for r in raw)
    scale = annual_total / raw_total if raw_total > 0 else 1.0
    return [r * scale for r in raw]


# ════════════════════════════════════════════════════════════════════════════
# SETTINGS — change these before running
# ════════════════════════════════════════════════════════════════════════════
ANNUAL_PRECIP_MM = 14.5    # Annual precipitation in mm (for RCNN simulation)
NUM_YEARS        = 100       # Number of years to simulate

# Equilibrium settings (must match training in train_rcnn_batch.py)
EQUILIBRIUM_PRECIP = 10.5   # Precipitation used to generate equilibrium state
EQUILIBRIUM_YEARS  = 100    # Years of PDE spin-up for equilibrium
GRID_SIZE          = (150, 150)
DATA_SEED          = 42     # Same seed as training for reproducible initial state
# ════════════════════════════════════════════════════════════════════════════

# Paths & fixed config
RETRAIN_DIR = "results/synthetic_rcnn_4site_retrain"
SUMMARY_JSON = os.path.join(RETRAIN_DIR, "summary_00-09.json")
SAVE_DIR = "results/synthetic_data/rcnn_simulation_results"

# Model config (must match what was used in train_rcnn_batch.py)
RCNN_CONFIG = {"hidden_channels": 16, "num_layers": 1, "kernel_size": 3}


# ── Helpers ─────────────────────────────────────────────────────────────────
def find_best_run() -> dict:
    """Find the run with the lowest final training loss."""
    with open(SUMMARY_JSON) as f:
        runs = json.load(f)
    best = min(runs, key=lambda r: r["final_loss"])
    print(f"Best RCNN run: run_{best['run_id']:02d}  "
          f"(loss = {best['final_loss']:.4f}, seed = {best['seed']})")
    return best


def load_model(run_id: int, device: torch.device) -> RCNNBaseline:
    """Load a trained RCNN model from a .pt checkpoint."""
    model = RCNNBaseline(**RCNN_CONFIG).to(device)
    pt_path = os.path.join(RETRAIN_DIR, f"run_{run_id:02d}.pt")
    checkpoint = torch.load(pt_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"  Loaded {pt_path}")
    return model


def generate_initial_conditions(device: torch.device) -> torch.Tensor:
    """Generate synthetic equilibrium biomass using the ground-truth PDE model.

    Uses the same seed, grid size, and equilibrium settings as training
    so the initial state is identical to what the RCNN was trained on.
    """
    print(f"Generating synthetic equilibrium state ...")
    print(f"  Grid: {GRID_SIZE[0]}x{GRID_SIZE[1]}, "
          f"precip: {EQUILIBRIUM_PRECIP}, years: {EQUILIBRIUM_YEARS}, seed: {DATA_SEED}")

    set_seed(DATA_SEED)
    data_gen = SyntheticDataGenerator(grid_size=GRID_SIZE, device=device)
    with torch.no_grad():
        initial_biomass = data_gen.generate_equilibrium_state(
            equilibrium_precipitation=EQUILIBRIUM_PRECIP,
            equilibrium_years=EQUILIBRIUM_YEARS,
        )

    print(f"  Shape: {tuple(initial_biomass.shape)}, "
          f"range: {initial_biomass.min().item():.1f} – {initial_biomass.max().item():.1f}")
    return initial_biomass


def simulate(
    model: RCNNBaseline,
    initial_biomass: torch.Tensor,
    annual_precip_mm: float,
    num_years: int,
    device: torch.device,
) -> dict:
    """
    Run the RCNN forward for *num_years* years at a constant precipitation.
    Each year: 52 weekly forward steps through the model.

    Returns dict with biomass snapshots (index 0 = initial state).
    """
    weekly_precip = generate_weekly_precipitation(annual_precip_mm)

    b = initial_biomass.clone()
    hidden = None

    snapshots = {"biomass": [b.squeeze().cpu().numpy()]}

    print(f"\nSimulating {num_years} years at {annual_precip_mm:.1f} mm/yr ...")
    with torch.no_grad():
        for yr in range(1, num_years + 1):
            b, hidden = model.simulate_year_weekly(b, weekly_precip, hidden)
            b = b.detach()
            hidden = [h.detach() for h in hidden]

            snapshots["biomass"].append(b.squeeze().cpu().numpy())

            if yr % max(1, num_years // 10) == 0 or yr == num_years:
                print(f"  Year {yr:>4d}  |  biomass mean={b.mean().item():.2f}  "
                      f"min={b.min().item():.2f}  max={b.max().item():.2f}")

    return snapshots


# ── Visualisation ───────────────────────────────────────────────────────────
def plot_results(snapshots: dict, annual_precip_mm: float, num_years: int,
                 run_id: int, save_dir: str):
    """Create and save summary figures."""
    biomass = np.array(snapshots["biomass"])
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
    ax.set_title(f"RCNN biomass evolution — run {run_id:02d}, {annual_precip_mm:.0f} mm/yr")
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

    for ax_idx in range(len(indices), nrows * ncols):
        r, c = divmod(ax_idx, ncols)
        axes[r, c].axis("off")

    fig.suptitle(
        f"RCNN biomass snapshots — run {run_id:02d}, {annual_precip_mm:.0f} mm/yr",
        fontsize=14,
    )
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, "biomass_snapshots.png"), dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    print("  Saved biomass_snapshots.png")

    # ── 3. Initial vs final comparison ─────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, (label, idx) in zip(axes, [("Initial (year 0)", 0), (f"Final (year {n_frames - 1})", -1)]):
        im = ax.imshow(biomass[idx], cmap="RdYlGn", norm=norm)
        ax.set_title(f"{label}\nmean={biomass[idx].mean():.2f}, "
                     f"min={biomass[idx].min():.2f}, max={biomass[idx].max():.2f}",
                     fontsize=11)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(
        f"RCNN initial vs final — run {run_id:02d}, {annual_precip_mm:.0f} mm/yr, {num_years} years",
        fontsize=14,
    )
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, "biomass_initial_vs_final.png"), dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    print("  Saved biomass_initial_vs_final.png")


# ── Main ────────────────────────────────────────────────────────────────────
#%%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

run_tag = f"precip{ANNUAL_PRECIP_MM:.0f}_years{NUM_YEARS}"
save_dir = os.path.join(SAVE_DIR, run_tag)
os.makedirs(save_dir, exist_ok=True)

# 1. Find and load best model
best_run = find_best_run()
run_id = best_run["run_id"]
model = load_model(run_id, device)

# 2. Generate synthetic equilibrium initial conditions
initial_biomass = generate_initial_conditions(device)

# 3. Simulate
snapshots = simulate(model, initial_biomass, ANNUAL_PRECIP_MM, NUM_YEARS, device)

# 4. Save config + numerical results
config = {
    "run_id": run_id,
    "annual_precip_mm": ANNUAL_PRECIP_MM,
    "num_years": NUM_YEARS,
    "rcnn_config": RCNN_CONFIG,
    "initial_conditions": "synthetic_equilibrium",
    "equilibrium_precipitation": EQUILIBRIUM_PRECIP,
    "equilibrium_years": EQUILIBRIUM_YEARS,
    "grid_size": list(GRID_SIZE),
    "data_seed": DATA_SEED,
    "final_training_loss": best_run["final_loss"],
}
with open(os.path.join(save_dir, "simulation_config.json"), "w") as f:
    json.dump(config, f, indent=2)

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
plot_results(snapshots, ANNUAL_PRECIP_MM, NUM_YEARS, run_id, save_dir)

print(f"\nAll outputs saved to {save_dir}/")

# %%
