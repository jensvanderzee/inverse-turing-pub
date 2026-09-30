# -*- coding: utf-8 -*-
"""
One-at-a-time (OAT) sensitivity analysis for the Rietkerk PDE model.

For each parameter:
  1. Run with ground-truth values → baseline biomass fields
  2. Perturb *one* parameter by a set of relative deltas (e.g. ±5 %, ±10 %, ±20 %)
  3. Re-run the model and compare the biomass output pixel-by-pixel to baseline
  4. Quantify sensitivity as the pixel-wise normalised RMSE

Outputs (saved to results/synthetic/sensitivity_analysis/):
  - sensitivity_summary.csv            per-parameter sensitivity indices
  - sensitivity_bar.png                bar chart ranking parameters
  - sensitivity_curves.png             sensitivity vs perturbation magnitude
  - elasticity.png                     dimensionless elasticity ranking
  - pixel_sensitivity_maps.png         per-pixel RMSE maps (3×3 grid)
  - biomass_timeseries_<param>.png     overlay of perturbed vs baseline time series
  - spatial_difference_<param>.png     maps of biomass difference at final time step
"""
# %%
import os
import random
import warnings
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

warnings.filterwarnings("ignore")

# ── Reproduce model classes from the training script ───────────────────────
# (kept self-contained so the script can run independently)


class EcologicalParameters:
    SURFACE_WATER_DIFFUSION = 8.0
    SOIL_WATER_DIFFUSION = 1.0
    BIOMASS_DIFFUSION = 0.05
    EVAPORATION_RATE = 0.3
    SEEPAGE_RATE = 0.4
    MORTALITY_RATE = 0.6
    INFILTRATION_RATE = 2.1
    PLANT_UPTAKE_RATE = 1.9
    BASE_PRECIPITATION = 0.675
    WATER_USE_EFFICIENCY = 0.55
    GROWTH_FACTOR_ETA = 0
    GROWTH_EXPONENT_Q = 0
    TIME_STEP = 0.02
    SAMPLE_INTERVAL = 50


class invRietkerk(nn.Module):
    """Forward Rietkerk PDE model (non-trainable version used for simulation)."""

    def __init__(self):
        super().__init__()
        self.time_step = EcologicalParameters.TIME_STEP
        self._setup_spatial_operators()

        # Default to ground-truth parameters
        self.infiltration_rate = EcologicalParameters.INFILTRATION_RATE
        self.evaporation_rate = EcologicalParameters.EVAPORATION_RATE
        self.seepage_rate = EcologicalParameters.SEEPAGE_RATE
        self.plant_uptake_rate = EcologicalParameters.PLANT_UPTAKE_RATE
        self.mortality_rate = EcologicalParameters.MORTALITY_RATE
        self.water_use_efficiency = EcologicalParameters.WATER_USE_EFFICIENCY
        self.surface_water_diffusion_coeff = EcologicalParameters.SURFACE_WATER_DIFFUSION
        self.soil_water_diffusion_coeff = EcologicalParameters.SOIL_WATER_DIFFUSION
        self.biomass_diffusion_coeff = EcologicalParameters.BIOMASS_DIFFUSION
        self.growth_factor_eta = EcologicalParameters.GROWTH_FACTOR_ETA
        self.growth_exponent_q = EcologicalParameters.GROWTH_EXPONENT_Q
        self.precipitation = 0.0

    def _setup_spatial_operators(self):
        laplacian_kernel = torch.tensor(
            [[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32
        )
        for name in ["surface_water", "soil_water", "biomass"]:
            conv = nn.Conv2d(1, 1, kernel_size=3, padding=1,
                             padding_mode="replicate", bias=False)
            conv.weight = nn.Parameter(
                laplacian_kernel[None, None, :], requires_grad=False
            )
            setattr(self, f"{name}_diffusion_op", conv)

    @torch.no_grad()
    def forward(self, surface_water, soil_water, biomass):
        growth_term = 1 + self.growth_factor_eta * biomass ** self.growth_exponent_q

        surface_water = surface_water + (
            self.surface_water_diffusion_coeff
            * self.surface_water_diffusion_op(surface_water)
            - self.evaporation_rate * surface_water
            + self.precipitation
            - self.infiltration_rate * surface_water * biomass
        ) * self.time_step

        soil_water = soil_water + (
            self.soil_water_diffusion_coeff
            * self.soil_water_diffusion_op(soil_water)
            - self.seepage_rate * soil_water
            + self.infiltration_rate * surface_water * biomass
            - self.plant_uptake_rate * soil_water * biomass * growth_term
        ) * self.time_step

        biomass = biomass + (
            self.biomass_diffusion_coeff * self.biomass_diffusion_op(biomass)
            - self.mortality_rate * biomass
            + self.water_use_efficiency
            * self.plant_uptake_rate
            * soil_water
            * biomass
            * growth_term
        ) * self.time_step

        return surface_water, soil_water, biomass


# ── Sensitivity-analysis helpers ───────────────────────────────────────────

GROUND_TRUTH: Dict[str, float] = {
    "infiltration_rate": EcologicalParameters.INFILTRATION_RATE,
    "seepage_rate": EcologicalParameters.SEEPAGE_RATE,
    "plant_uptake_rate": EcologicalParameters.PLANT_UPTAKE_RATE,
    "mortality_rate": EcologicalParameters.MORTALITY_RATE,
    "evaporation_rate": EcologicalParameters.EVAPORATION_RATE,
    "water_use_efficiency": EcologicalParameters.WATER_USE_EFFICIENCY,
    "surface_water_diffusion_coeff": EcologicalParameters.SURFACE_WATER_DIFFUSION,
    "soil_water_diffusion_coeff": EcologicalParameters.SOIL_WATER_DIFFUSION,
    "biomass_diffusion_coeff": EcologicalParameters.BIOMASS_DIFFUSION,
}

PARAM_LABELS = {
    "infiltration_rate": "Infiltration",
    "seepage_rate": "Seepage",
    "plant_uptake_rate": "Plant uptake",
    "mortality_rate": "Mortality",
    "evaporation_rate": "Evaporation",
    "water_use_efficiency": "WUE",
    "surface_water_diffusion_coeff": "Surf. water diff.",
    "soil_water_diffusion_coeff": "Soil water diff.",
    "biomass_diffusion_coeff": "Biomass diff.",
}

# Perturbation magnitudes (relative)
PERTURBATIONS = [-0.20, -0.10, -0.05, 0.05, 0.10, 0.20]


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def build_model(device: torch.device, overrides: Dict[str, float] = None):
    """Create a fresh model with ground-truth params, optionally overriding some."""
    model = invRietkerk().to(device)
    for pname, pval in GROUND_TRUTH.items():
        setattr(model, pname, pval)
    if overrides:
        for pname, pval in overrides.items():
            setattr(model, pname, pval)
    return model


def generate_equilibrium(model: invRietkerk, device: torch.device,
                         precip: float = 43.0, steps: int = 2500):
    """Spin-up the model to reach spatial equilibrium."""
    sample_interval = EcologicalParameters.SAMPLE_INTERVAL
    sw = torch.rand(1, 1, 150, 150, device=device)
    slw = torch.rand(1, 1, 150, 150, device=device)
    bio = torch.rand(1, 1, 150, 150, device=device)

    model.precipitation = 0.0
    for step in range(steps):
        if step % sample_interval == 0:
            model.precipitation = precip
        sw, slw, bio = model.forward(sw, slw, bio)
        model.precipitation = 0.0

    return bio.clone()


def run_paired_pixelwise(
    baseline_model: invRietkerk,
    perturbed_model: invRietkerk,
    equilibrium_state: torch.Tensor,
    precipitation_values: List[float],
    time_series_steps: int,
    device: torch.device,
) -> Dict[str, object]:
    """
    Run baseline and perturbed models in lockstep, accumulating pixel-wise
    squared differences on the GPU without storing all intermediate fields.

    Returns dict with:
      - 'nrmse'              : scalar — pixel-wise NRMSE over all pixels,
                               timesteps and sites
      - 'pixel_sensitivity'  : (150, 150) array — per-pixel RMSE averaged
                               across timesteps and sites
      - 'mean_biomass_base'  : list — spatial-mean baseline biomass per
                               sample step (site 0, for time-series plots)
      - 'mean_biomass_pert'  : list — same for the perturbed run
      - 'final_biomass_base' : (150, 150) — baseline biomass at last step
                               (site 0)
      - 'final_biomass_pert' : (150, 150) — perturbed biomass at last step
                               (site 0)
    """
    sample_interval = EcologicalParameters.SAMPLE_INTERVAL
    H, W = 150, 150

    # Accumulators for pixel-wise comparison
    pixel_sse_sum = torch.zeros(H, W, device=device)
    pixel_base_abs_sum = torch.zeros(H, W, device=device)
    n_samples_total = 0

    # Time-series and spatial caches (site 0 only)
    mean_base_ts: List[float] = []
    mean_pert_ts: List[float] = []
    final_base = None
    final_pert = None

    for site_idx, precip in enumerate(precipitation_values):
        # Identical initial conditions for both runs
        sw_b  = torch.zeros(1, 1, H, W, device=device)
        slw_b = torch.zeros(1, 1, H, W, device=device)
        bio_b = equilibrium_state.clone()

        sw_p  = torch.zeros(1, 1, H, W, device=device)
        slw_p = torch.zeros(1, 1, H, W, device=device)
        bio_p = equilibrium_state.clone()

        site_pixel_sse = torch.zeros(H, W, device=device)
        site_pixel_base = torch.zeros(H, W, device=device)

        for step in range(time_series_steps):
            p = precip if (step % sample_interval == 0) else 0.0
            baseline_model.precipitation = p
            perturbed_model.precipitation = p

            sw_b, slw_b, bio_b = baseline_model.forward(sw_b, slw_b, bio_b)
            sw_p, slw_p, bio_p = perturbed_model.forward(sw_p, slw_p, bio_p)

            if step % sample_interval == 0 and step > 0:
                diff = (bio_p - bio_b).squeeze()
                site_pixel_sse += diff ** 2
                site_pixel_base += bio_b.squeeze().abs()
                n_samples_total += 1

                # Track mean-biomass time series for site 0
                if site_idx == 0:
                    mean_base_ts.append(bio_b.mean().item())
                    mean_pert_ts.append(bio_p.mean().item())

        pixel_sse_sum += site_pixel_sse
        pixel_base_abs_sum += site_pixel_base

        # Cache final fields for site 0
        if site_idx == 0:
            final_base = bio_b.squeeze().cpu().numpy()
            final_pert = bio_p.squeeze().cpu().numpy()

    # ── Aggregate ──────────────────────────────────────────────────────────
    pixel_rmse = torch.sqrt(pixel_sse_sum / n_samples_total).cpu().numpy()

    total_mse = (pixel_sse_sum.sum() / n_samples_total).item() / (H * W)
    total_rmse = np.sqrt(total_mse)
    mean_abs_base = (pixel_base_abs_sum.sum()
                     / (n_samples_total * H * W)).item()
    nrmse = total_rmse / mean_abs_base if mean_abs_base > 1e-12 else 0.0

    return {
        "nrmse": nrmse,
        "pixel_sensitivity": pixel_rmse,
        "mean_biomass_base": mean_base_ts,
        "mean_biomass_pert": mean_pert_ts,
        "final_biomass_base": final_base,
        "final_biomass_pert": final_pert,
    }


# ── Plotting helpers ───────────────────────────────────────────────────────

def plot_sensitivity_bar(results_df: pd.DataFrame, save_path: str):
    """Rank parameters by sensitivity at the reference perturbation (+10 %)."""
    ref = results_df[results_df["perturbation"] == 0.10].copy()
    if ref.empty:
        ref = results_df.groupby("parameter").agg({"nrmse": "max"}).reset_index()
    ref = ref.sort_values("nrmse", ascending=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    labels = [PARAM_LABELS.get(p, p) for p in ref["parameter"]]
    bars = ax.barh(labels, ref["nrmse"], color="steelblue", edgecolor="white")
    ax.set_xlabel("Pixel-wise Normalised RMSE (+10 % perturbation)", fontsize=12)
    ax.set_title("Parameter Sensitivity Ranking", fontsize=14, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    for bar, val in zip(bars, ref["nrmse"]):
        ax.text(val + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "sensitivity_bar.png"), dpi=300,
                bbox_inches="tight")
    plt.close()


def plot_sensitivity_curves(results_df: pd.DataFrame, save_path: str):
    """Pixel-wise NRMSE vs perturbation magnitude for every parameter."""
    fig, ax = plt.subplots(figsize=(10, 6))
    cmap = plt.cm.tab10
    params = results_df["parameter"].unique()
    for i, param in enumerate(params):
        sub = results_df[results_df["parameter"] == param].sort_values("perturbation")
        ax.plot(sub["perturbation"] * 100, sub["nrmse"],
                marker="o", label=PARAM_LABELS.get(param, param),
                color=cmap(i / len(params)), linewidth=1.5)
    ax.axvline(0, color="gray", linestyle="--", linewidth=0.5)
    ax.set_xlabel("Perturbation (%)", fontsize=12)
    ax.set_ylabel("Pixel-wise Normalised RMSE", fontsize=12)
    ax.set_title("Sensitivity Curves", fontsize=14, fontweight="bold")
    ax.legend(fontsize=8, ncol=2, loc="upper left")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "sensitivity_curves.png"), dpi=300,
                bbox_inches="tight")
    plt.close()


def plot_biomass_timeseries(param_name: str,
                            baseline_means: List[float],
                            perturbed_runs: Dict[float, List[float]],
                            save_path: str):
    """Overlay baseline and perturbed mean-biomass time series for one parameter."""
    fig, ax = plt.subplots(figsize=(10, 5))
    t = np.arange(len(baseline_means))
    ax.plot(t, baseline_means, "k-", linewidth=2, label="Baseline")
    cmap = plt.cm.coolwarm
    deltas = sorted(perturbed_runs.keys())
    norm = plt.Normalize(vmin=min(deltas), vmax=max(deltas))
    for delta in deltas:
        ax.plot(t, perturbed_runs[delta], linewidth=1.2,
                color=cmap(norm(delta)),
                label=f"{delta:+.0%}")
    ax.set_xlabel("Sample step")
    ax.set_ylabel("Mean biomass")
    ax.set_title(f"Biomass response to perturbation of "
                 f"{PARAM_LABELS.get(param_name, param_name)}",
                 fontweight="bold")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"biomass_timeseries_{param_name}.png"),
                dpi=200, bbox_inches="tight")
    plt.close()


def plot_spatial_difference(param_name: str,
                            baseline_field: np.ndarray,
                            perturbed_fields: Dict[float, np.ndarray],
                            save_path: str):
    """Show spatial maps of biomass difference at final time step."""
    deltas = sorted(perturbed_fields.keys())
    n = len(deltas)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]

    vmax = max(np.abs(perturbed_fields[d] - baseline_field).max() for d in deltas)
    vmax = max(vmax, 1e-6)

    for ax, delta in zip(axes, deltas):
        diff = perturbed_fields[delta] - baseline_field
        im = ax.imshow(diff, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_title(f"{delta:+.0%}", fontsize=11)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(f"Biomass difference (perturbed − baseline)\n"
                 f"Parameter: {PARAM_LABELS.get(param_name, param_name)}",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"spatial_difference_{param_name}.png"),
                dpi=200, bbox_inches="tight")
    plt.close()


def plot_pixel_sensitivity_maps(pixel_maps: Dict[str, np.ndarray],
                                 save_path: str):
    """3×3 grid of per-pixel RMSE maps (one per parameter, at +10 % perturbation)."""
    params = list(pixel_maps.keys())
    n = len(params)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows))
    axes = np.array(axes).flatten()

    for i, param in enumerate(params):
        ax = axes[i]
        im = ax.imshow(pixel_maps[param], cmap="inferno")
        ax.set_title(PARAM_LABELS.get(param, param), fontsize=12,
                     fontweight="bold")
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    fig.suptitle("Per-pixel sensitivity (RMSE over time)\n+10 % perturbation",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "pixel_sensitivity_maps.png"),
                dpi=200, bbox_inches="tight")
    plt.close()


def plot_elasticity(results_df: pd.DataFrame, save_path: str):
    """
    Elasticity = (Δoutput / output) / (Δparameter / parameter).
    Estimated from small perturbations (±5 %).  A dimensionless measure
    of proportional sensitivity that is comparable across parameters.
    """
    small = results_df[results_df["perturbation"].abs() <= 0.05].copy()
    if small.empty:
        return

    small["elasticity"] = small["nrmse"] / small["perturbation"].abs()
    elas = small.groupby("parameter")["elasticity"].mean().sort_values()

    fig, ax = plt.subplots(figsize=(8, 6))
    labels = [PARAM_LABELS.get(p, p) for p in elas.index]
    ax.barh(labels, elas.values, color="darkorange", edgecolor="white")
    ax.set_xlabel("Elasticity (dimensionless)", fontsize=12)
    ax.set_title("Parameter Elasticity\n"
                 "(proportional output change per proportional input change)",
                 fontsize=13, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    for i, (lbl, val) in enumerate(zip(labels, elas.values)):
        ax.text(val + 0.005, i, f"{val:.3f}", va="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "elasticity.png"), dpi=300,
                bbox_inches="tight")
    plt.close()


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    save_dir = os.path.join("results", "synthetic", "sensitivity_analysis")
    os.makedirs(save_dir, exist_ok=True)

    # -- 1. Generate equilibrium with ground-truth model ────────────────────
    print("Generating equilibrium state …")
    gt_model = build_model(device)
    equilibrium = generate_equilibrium(gt_model, device,
                                       precip=43.0, steps=2500)

    # Simulation settings
    precipitation_values = [43.0, 48.0, 53.0]  # 3 precipitation levels
    time_series_steps = 501

    # -- 2. Perturbed runs (paired pixel-wise) ──────────────────────────────
    rows = []
    # Caches for per-parameter plots (site 0)
    baseline_mean_ts: List[float] = []       # set once from first run
    ts_cache: Dict[str, Dict[float, List[float]]] = {}
    field_cache_base: Dict[str, np.ndarray] = {}
    field_cache_pert: Dict[str, Dict[float, np.ndarray]] = {}
    pixel_maps_10pct: Dict[str, np.ndarray] = {}

    for param_name, gt_val in GROUND_TRUTH.items():
        print(f"  Perturbing {param_name} (GT = {gt_val}) …")
        ts_cache[param_name] = {}
        field_cache_pert[param_name] = {}

        for delta in PERTURBATIONS:
            new_val = gt_val * (1.0 + delta)
            perturbed_model = build_model(device,
                                          overrides={param_name: new_val})

            pw = run_paired_pixelwise(
                gt_model, perturbed_model, equilibrium,
                precipitation_values, time_series_steps, device,
            )

            rows.append({
                "parameter": param_name,
                "gt_value": gt_val,
                "perturbation": delta,
                "new_value": new_val,
                "nrmse": pw["nrmse"],
            })

            # Cache time series and fields for plots (site 0)
            ts_cache[param_name][delta] = pw["mean_biomass_pert"]
            field_cache_pert[param_name][delta] = pw["final_biomass_pert"]

            # Baseline is the same for every delta; grab it once
            if not baseline_mean_ts:
                baseline_mean_ts = pw["mean_biomass_base"]
            if param_name not in field_cache_base:
                field_cache_base[param_name] = pw["final_biomass_base"]

            if abs(delta - 0.10) < 1e-9:
                pixel_maps_10pct[param_name] = pw["pixel_sensitivity"]

    results_df = pd.DataFrame(rows)
    results_df.to_csv(os.path.join(save_dir, "sensitivity_summary.csv"),
                      index=False)
    print(f"\nSaved sensitivity_summary.csv  ({len(results_df)} runs)")

    # -- 3. Summary table ───────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print("SENSITIVITY SUMMARY (pixel-wise NRMSE at ±10 % perturbation)")
    print(f"{'='*72}")
    ref = results_df[results_df["perturbation"].isin([0.10, -0.10])]
    summary = ref.groupby("parameter")["nrmse"].mean().sort_values(ascending=False)
    for p, v in summary.items():
        print(f"  {PARAM_LABELS.get(p, p):>20s}   NRMSE = {v:.6f}")

    # -- 4. Plots ───────────────────────────────────────────────────────────
    print("\nGenerating plots …")
    plot_sensitivity_bar(results_df, save_dir)
    plot_sensitivity_curves(results_df, save_dir)
    plot_elasticity(results_df, save_dir)

    if pixel_maps_10pct:
        plot_pixel_sensitivity_maps(pixel_maps_10pct, save_dir)

    for param_name in GROUND_TRUTH:
        plot_biomass_timeseries(param_name, baseline_mean_ts,
                                ts_cache[param_name], save_dir)
        big_deltas = {d: field_cache_pert[param_name][d]
                      for d in [-0.20, -0.10, 0.10, 0.20]
                      if d in field_cache_pert[param_name]}
        if big_deltas:
            plot_spatial_difference(param_name, field_cache_base[param_name],
                                   big_deltas, save_dir)

    print(f"\nAll outputs saved to:  {save_dir}/")


if __name__ == "__main__":
    main()
