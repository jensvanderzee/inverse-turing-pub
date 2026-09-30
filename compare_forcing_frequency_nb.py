# -*- coding: utf-8 -*-
"""
Compare the effect of rainfall forcing frequency on the Rietkerk PDE simulation.

Runs three scenarios with identical total annual precipitation but different
temporal distributions:
  - Daily:  precipitation spread uniformly across 365 days
  - Weekly: precipitation spread uniformly across 52 weeks
  - Annual: all precipitation applied in a single pulse event
"""

# %%
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import os
from typing import Tuple, Dict

# ---------------------------------------------------------------------------
# ⚙️  USER PARAMETERS — edit these to configure your experiment
# ---------------------------------------------------------------------------

YEARS              = 10
ANNUAL_PRECIP      = 300.0    # mm/yr
GRID_SIZE          = 150      # GRID_SIZE × GRID_SIZE
SPIN_UP_YEARS      = 5
SNAPSHOTS_PER_YEAR = 12

SAVE_DIR = "./results/forcing_comparison"

# ---------------------------------------------------------------------------
# Model parameters (ground truth)
# ---------------------------------------------------------------------------

# %%
class EcologicalParameters:
    SURFACE_WATER_DIFFUSION = 8.0
    SOIL_WATER_DIFFUSION    = 1.0
    BIOMASS_DIFFUSION       = 0.05

    EVAPORATION_RATE  = 0.3
    SEEPAGE_RATE      = 0.4
    MORTALITY_RATE    = 0.6

    INFILTRATION_RATE = 2.1
    PLANT_UPTAKE_RATE = 1.9

    BASE_PRECIPITATION   = 0.675
    WATER_USE_EFFICIENCY = 0.55
    GROWTH_FACTOR_ETA    = 0
    GROWTH_EXPONENT_Q    = 0
    TIME_STEP            = 0.01   # base dt (days) — shrink this to improve stability


# Derived constant — used everywhere steps need to map to real days.
# Automatically updates when TIME_STEP changes.
STEPS_PER_DAY = round(1.0 / EcologicalParameters.TIME_STEP)

print(f"TIME_STEP     = {EcologicalParameters.TIME_STEP}")
print(f"STEPS_PER_DAY = {STEPS_PER_DAY}")


# ---------------------------------------------------------------------------
# Rietkerk PDE model (forward-Euler, non-trainable ground truth)
# ---------------------------------------------------------------------------

# %%
class Rietkerk(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_time_step = EcologicalParameters.TIME_STEP
        self._setup_spatial_operators()
        self._set_ground_truth_params()

    def _setup_spatial_operators(self):
        laplacian_kernel = torch.tensor(
            [[0, 1, 0],
             [1, -4, 1],
             [0, 1, 0]], dtype=torch.float32
        )
        for name in ['surface_water', 'soil_water', 'biomass']:
            conv = nn.Conv2d(1, 1, kernel_size=3, padding=1,
                             padding_mode="replicate", bias=False)
            conv.weight = nn.Parameter(laplacian_kernel[None, None, :],
                                       requires_grad=False)
            setattr(self, f"{name}_diffusion_op", conv)

    def _set_ground_truth_params(self):
        p = EcologicalParameters
        self.infiltration_rate       = p.INFILTRATION_RATE
        self.evaporation_rate        = p.EVAPORATION_RATE
        self.seepage_rate            = p.SEEPAGE_RATE
        self.plant_uptake_rate       = p.PLANT_UPTAKE_RATE
        self.mortality_rate          = p.MORTALITY_RATE
        self.water_use_efficiency    = p.WATER_USE_EFFICIENCY
        self.growth_factor_eta       = p.GROWTH_FACTOR_ETA
        self.growth_exponent_q       = p.GROWTH_EXPONENT_Q
        self.surface_water_diffusion = p.SURFACE_WATER_DIFFUSION
        self.soil_water_diffusion    = p.SOIL_WATER_DIFFUSION
        self.biomass_diffusion       = p.BIOMASS_DIFFUSION

    @torch.no_grad()
    def forward(self, S: torch.Tensor, W: torch.Tensor, B: torch.Tensor,
                precip: float = 0.0, dt: float = None,
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if dt is None:
            dt = self.base_time_step

        growth = 1 + self.growth_factor_eta * B ** self.growth_exponent_q

        S = S + (self.surface_water_diffusion * self.surface_water_diffusion_op(S)
                 - self.evaporation_rate * S
                 + precip
                 - self.infiltration_rate * S * B) * dt

        W = W + (self.soil_water_diffusion * self.soil_water_diffusion_op(W)
                 - self.seepage_rate * W
                 + self.infiltration_rate * S * B
                 - self.plant_uptake_rate * W * B * growth) * dt

        B = B + (self.biomass_diffusion * self.biomass_diffusion_op(B)
                 - self.mortality_rate * B
                 + self.water_use_efficiency * self.plant_uptake_rate
                   * W * B * growth) * dt

        return S, W, B


# ---------------------------------------------------------------------------
# Simulation helpers
# ---------------------------------------------------------------------------

# %%
def make_initial_state(grid_size: int, device: torch.device):
    """Random initial conditions in [0, 1]."""
    shape = (1, 1, grid_size, grid_size)
    S = torch.rand(shape, device=device)
    W = torch.rand(shape, device=device)
    B = torch.rand(shape, device=device)
    return S, W, B


def spin_up(model: Rietkerk, S, W, B, annual_precip: float,
            spin_up_years: int = 5) -> Tuple[torch.Tensor, ...]:
    """
    Spin up the model to approximate equilibrium using daily forcing.
    Steps per year = 365 * STEPS_PER_DAY so real time is always correct
    regardless of TIME_STEP.
    """
    daily_rate     = annual_precip / 365.0
    steps_per_year = 365 * STEPS_PER_DAY

    for _ in range(spin_up_years):
        for _ in range(steps_per_year):
            S, W, B = model.forward(S, W, B, precip=daily_rate)
    return S, W, B


# ---------------------------------------------------------------------------
# Three forcing strategies
# ---------------------------------------------------------------------------

# %%
def simulate_daily(model: Rietkerk, S, W, B, annual_precip: float,
                   n_years: int, snapshots_per_year: int = 12):
    """Precipitation spread uniformly across 365 days."""
    daily_rate        = annual_precip / 365.0
    steps_per_year    = 365 * STEPS_PER_DAY
    snapshot_interval = max(1, steps_per_year // snapshots_per_year)

    history = _record_snapshot([], S, W, B, year=0, step=0)

    for yr in range(n_years):
        for step in range(steps_per_year):
            S, W, B = model.forward(S, W, B, precip=daily_rate)
            if (step + 1) % snapshot_interval == 0:
                history = _record_snapshot(history, S, W, B,
                                           year=yr, step=step + 1)
    return history


def simulate_weekly(model: Rietkerk, S, W, B, annual_precip: float,
                    n_years: int, snapshots_per_year: int = 12):
    """
    Precipitation applied in one pulse at the start of each week,
    then zero for the remaining 6 days.
    The pulse is expressed as a rate over one time-step so that
    pulse_rate * dt = weekly_total (amount is conserved regardless of dt).
    """
    weekly_total      = annual_precip / 52.0
    pulse_rate        = weekly_total / model.base_time_step
    steps_per_week    = 7 * STEPS_PER_DAY
    steps_per_year    = 52 * steps_per_week
    snapshot_interval = max(1, steps_per_year // snapshots_per_year)

    history      = _record_snapshot([], S, W, B, year=0, step=0)
    step_in_year = 0

    for yr in range(n_years):
        step_in_year = 0
        for week in range(52):
            for s in range(steps_per_week):
                # Pulse only on the very first sub-step of each week
                p = pulse_rate if s == 0 else 0.0
                S, W, B = model.forward(S, W, B, precip=p)
                step_in_year += 1
                if step_in_year % snapshot_interval == 0:
                    history = _record_snapshot(history, S, W, B,
                                               year=yr, step=step_in_year)
    return history


def simulate_annual(model: Rietkerk, S, W, B, annual_precip: float,
                    n_years: int, snapshots_per_year: int = 12,
                    rain_day: int = 91, fine_period_days: int = 30,
                    fine_factor: int = 10):
    """
    All precipitation applied in one pulse on rain_day.
    Uses a finer sub-step (dt / fine_factor) for fine_period_days after
    the pulse to resolve the sharp transient, then reverts to base dt.
    The pulse rate is scaled to fine_dt so the total added water = annual_precip.
    """
    dt      = model.base_time_step
    fine_dt = dt / fine_factor

    # Express time boundaries in coarse steps
    steps_per_year    = 365 * STEPS_PER_DAY
    rain_step         = rain_day * STEPS_PER_DAY          # first coarse step of rain_day
    fine_end_step     = rain_step + fine_period_days * STEPS_PER_DAY * fine_factor
    snapshot_interval = max(1, steps_per_year // snapshots_per_year)

    history = _record_snapshot([], S, W, B, year=0, step=0)

    for yr in range(n_years):
        step = 0
        while step < steps_per_year:
            in_fine_period = (rain_step <= step < fine_end_step)

            if in_fine_period:
                # Rate scaled to fine_dt so total water added = annual_precip
                p = (annual_precip / fine_dt) if step == rain_step else 0.0
                S, W, B = model.forward(S, W, B, precip=p, dt=fine_dt)
                step += 1
            else:
                S, W, B = model.forward(S, W, B, precip=0.0, dt=dt)
                step += 1

            if step % snapshot_interval == 0:
                history = _record_snapshot(history, S, W, B,
                                           year=yr, step=step)

    return history


# ---------------------------------------------------------------------------
# Recording / plotting
# ---------------------------------------------------------------------------

# %%
def _record_snapshot(history: list, S, W, B, year: int, step: int):
    history.append({
        'year':               year,
        'step':               step,
        'surface_water_mean': S.mean().item(),
        'soil_water_mean':    W.mean().item(),
        'biomass_mean':       B.mean().item(),
        'biomass_std':        B.std().item(),
        'biomass_snapshot':   B[0, 0].cpu().numpy().copy(),
    })
    return history


def plot_comparison(results: Dict[str, list], annual_precip: float,
                    n_years: int, save_dir: str):
    """Create a multi-panel figure comparing forcing strategies."""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"Rietkerk PDE – Rainfall forcing comparison\n"
        f"Annual precip = {annual_precip:.0f} mm,  {n_years} years  "
        f"(dt = {EcologicalParameters.TIME_STEP})",
        fontsize=13, fontweight='bold',
    )

    colours = {'daily': '#2196F3', 'weekly': '#FF9800', 'annual': '#E91E63'}

    # Panel 1: mean biomass
    ax = axes[0, 0]
    for label, hist in results.items():
        t       = np.arange(len(hist))
        biomass = [h['biomass_mean'] for h in hist]
        ax.plot(t, biomass, label=label, color=colours[label], linewidth=1.2)
    ax.set_ylabel('Mean biomass')
    ax.set_xlabel('Snapshot index')
    ax.set_title('Mean biomass over time')
    ax.legend(); ax.grid(True, alpha=0.3)

    # Panel 2: biomass std (spatial heterogeneity)
    ax = axes[0, 1]
    for label, hist in results.items():
        t   = np.arange(len(hist))
        std = [h['biomass_std'] for h in hist]
        ax.plot(t, std, label=label, color=colours[label], linewidth=1.2)
    ax.set_ylabel('Biomass spatial std')
    ax.set_xlabel('Snapshot index')
    ax.set_title('Spatial heterogeneity')
    ax.legend(); ax.grid(True, alpha=0.3)

    # Panel 3: mean surface water
    ax = axes[1, 0]
    for label, hist in results.items():
        t  = np.arange(len(hist))
        sw = [h['surface_water_mean'] for h in hist]
        ax.plot(t, sw, label=label, color=colours[label], linewidth=1.2)
    ax.set_ylabel('Mean surface water')
    ax.set_xlabel('Snapshot index')
    ax.set_title('Surface water')
    ax.legend(); ax.grid(True, alpha=0.3)

    # Panel 4: mean soil water
    ax = axes[1, 1]
    for label, hist in results.items():
        t  = np.arange(len(hist))
        sw = [h['soil_water_mean'] for h in hist]
        ax.plot(t, sw, label=label, color=colours[label], linewidth=1.2)
    ax.set_ylabel('Mean soil water')
    ax.set_xlabel('Snapshot index')
    ax.set_title('Soil water')
    ax.legend(); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, 'forcing_comparison_timeseries.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved: {path}")

    # Spatial snapshots at final timestep
    fig2, axes2 = plt.subplots(1, 3, figsize=(15, 4.5))
    fig2.suptitle('Final biomass field', fontsize=13, fontweight='bold')
    for i, (label, hist) in enumerate(results.items()):
        im = axes2[i].imshow(hist[-1]['biomass_snapshot'], cmap='YlGn', vmin=0)
        axes2[i].set_title(f'{label} forcing')
        axes2[i].axis('off')
        plt.colorbar(im, ax=axes2[i], fraction=0.046, pad=0.04)
    plt.tight_layout()
    path2 = os.path.join(save_dir, 'forcing_comparison_spatial.png')
    plt.savefig(path2, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved: {path2}")


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

# %%  Setup
os.makedirs(SAVE_DIR, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device : {device}")
print(f"Grid   : {GRID_SIZE}×{GRID_SIZE}")
print(f"Precip : {ANNUAL_PRECIP} mm/yr")
print(f"Years  : {YEARS}  (+ {SPIN_UP_YEARS} spin-up)")
print(f"dt     : {EcologicalParameters.TIME_STEP}  ({STEPS_PER_DAY} steps/day)")

model = Rietkerk().to(device)

# %%  Spin-up — identical starting point for all three scenarios
S0, W0, B0 = make_initial_state(GRID_SIZE, device)
S0, W0, B0 = spin_up(model, S0, W0, B0, ANNUAL_PRECIP, SPIN_UP_YEARS)
print("Spin-up complete.\n")

# %%  Run the three scenarios
scenarios = {
    'daily':  simulate_daily,
    'weekly': simulate_weekly,
    'annual': simulate_annual,
}

results: Dict[str, list] = {}
for name, sim_fn in scenarios.items():
    print(f"Running {name} forcing ...")
    history = sim_fn(
        model,
        S0.clone(), W0.clone(), B0.clone(),
        annual_precip=ANNUAL_PRECIP,
        n_years=YEARS,
        snapshots_per_year=SNAPSHOTS_PER_YEAR,
    )
    results[name] = history
    print(f"  -> final mean biomass = {history[-1]['biomass_mean']:.4f}  "
          f"({len(history)} snapshots)\n")

# %%  Plot
plot_comparison(results, ANNUAL_PRECIP, YEARS, SAVE_DIR)

# %%  Numerical summary
print("\n" + "=" * 55)
print(f"{'Forcing':<10} {'Final biomass':>15} {'Final Δ vs daily':>18}")
print("-" * 55)
daily_bm = results['daily'][-1]['biomass_mean']
for name, hist in results.items():
    bm    = hist[-1]['biomass_mean']
    delta = ((bm - daily_bm) / daily_bm * 100) if daily_bm != 0 else 0.0
    print(f"{name:<10} {bm:>15.4f} {delta:>17.1f}%")
print("=" * 55)
# %%
