#!/usr/bin/env python3
"""
Batch RCNN training script — run a range of training runs on a single GPU.

Derived from interactive_RCNN_weekly notebook with all visualisation,
pause/resume, and matplotlib code removed.

Usage
-----
# Launch 3 processes in parallel, each on a different GPU:
    ./launch.sh 0  9  0   # runs 0–9  on GPU 0
    ./launch.sh 10 19 1   # runs 10–19 on GPU 1
    ./launch.sh 20 29 2   # runs 20–29 on GPU 2

# Or call the Python script directly:
    python train_rcnn_batch.py --start 0 --end 9 --gpu 0
"""

import argparse
import json
import math
import os
import random
import time
from typing import List

import numpy as np
import torch
import torch.nn as nn

# ══════════════════════════════════════════════════════════════════════
#  0.  Reproducibility
# ══════════════════════════════════════════════════════════════════════
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ══════════════════════════════════════════════════════════════════════
#  1.  Ground-truth PDE model & data generation
# ══════════════════════════════════════════════════════════════════════
class EcologicalParameters:
    SURFACE_WATER_DIFFUSION = 8.0
    SOIL_WATER_DIFFUSION = 1.0
    BIOMASS_DIFFUSION = 0.05
    EVAPORATION_RATE = 0.6
    SEEPAGE_RATE = 0.8
    MORTALITY_RATE = 0.6
    INFILTRATION_RATE = 0.2
    PLANT_UPTAKE_RATE = 0.35
    BASE_PRECIPITATION = 0.675
    WATER_USE_EFFICIENCY = 0.35

    TIME_STEP = 1.0
    NOISE_LEVEL = 0.05
    STEPS_PER_WEEK = 1
    NUM_WEEKS = 52


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
    def __init__(self, grid_size=(128, 128), device=None):
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

    def generate_training_data(self, weekly_precipitation_profiles, equilibrium_state,
                               time_series_years=10, noise_level=0.02, steps_per_week=1):
        training_data = []
        for weekly_precip in weekly_precipitation_profiles:
            surface_water = torch.zeros(1, 1, self.height, self.width, device=self.device)
            soil_water = torch.zeros(1, 1, self.height, self.width, device=self.device)
            biomass = equilibrium_state.clone()
            site_ts = []
            for _ in range(time_series_years):
                surface_water, soil_water, biomass = self.ground_truth_model.simulate_year_weekly(
                    surface_water, soil_water, biomass,
                    weekly_precipitation=weekly_precip, steps_per_week=steps_per_week,
                )
                noisy_biomass = biomass + torch.randn_like(biomass) * noise_level
                site_ts.append(noisy_biomass.clone())
            training_data.append(site_ts)
        return training_data


# ══════════════════════════════════════════════════════════════════════
#  2.  ConvRNN cell
# ══════════════════════════════════════════════════════════════════════
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


# ══════════════════════════════════════════════════════════════════════
#  3.  Full RCNN model
# ══════════════════════════════════════════════════════════════════════
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


# ══════════════════════════════════════════════════════════════════════
#  4.  Single training run
# ══════════════════════════════════════════════════════════════════════
def train_single_run(
    run_id: int,
    seed: int,
    device: torch.device,
    equilibrium_state: torch.Tensor,
    training_data: list,
    weekly_precipitation_profiles: list,
    num_epochs: int = 20,
    gradient_clip: float = 1.0,
    print_every: int = 50,
) -> dict:
    """Train one RCNN from scratch and return results dict."""

    t0 = time.time()
    print(f"\n{'='*60}")
    print(f"  Run {run_id:02d}  |  seed={seed}  |  {num_epochs} epochs")
    print(f"{'='*60}")

    # — init model with this run's seed —
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = RCNNBaseline(hidden_channels=32, num_layers=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.99)
    loss_fn = nn.MSELoss()
    loss_history = []

    model.train()
    for epoch in range(num_epochs):
        optimizer.zero_grad()

        total_loss = 0.0
        for site_idx, weekly_precip in enumerate(weekly_precipitation_profiles):
            pred_biomass = equilibrium_state.clone()
            hidden = None
            prev_pred = pred_biomass.clone()
            prev_target = pred_biomass.clone()

            for year_idx in range(len(training_data[site_idx])):
                pred_biomass, hidden = model.simulate_year_weekly(
                    pred_biomass, weekly_precip, hidden,
                )
                hidden = [h.detach() for h in hidden]
                target = training_data[site_idx][year_idx]
                total_loss = total_loss + loss_fn(
                    pred_biomass - prev_pred,
                    target - prev_target,
                )
                prev_pred = pred_biomass.detach()
                prev_target = target

        total_loss.backward()

        if gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip)

        optimizer.step()
        scheduler.step()

        loss_val = total_loss.item()
        loss_history.append(loss_val)

        if math.isnan(loss_val):
            print(f"  [run {run_id:02d}] NaN loss at epoch {epoch} — stopping early.")
            break

        if epoch % print_every == 0 or epoch == num_epochs - 1:
            print(f"  [run {run_id:02d}] Epoch {epoch:4d}  Loss: {loss_val:.6f}  "
                  f"LR: {scheduler.get_last_lr()[0]:.2e}", flush=True)

    elapsed = time.time() - t0
    final_loss = loss_history[-1] if loss_history else float("nan")
    print(f"  [run {run_id:02d}] Done in {elapsed:.1f}s  |  "
          f"final loss: {final_loss:.6f}  |  epochs: {len(loss_history)}")

    return {
        "run_id": run_id,
        "seed": seed,
        "num_epochs": len(loss_history),
        "final_loss": final_loss,
        "loss_history": loss_history,
        "elapsed_seconds": elapsed,
        "model_state_dict": model.state_dict(),
    }


# ══════════════════════════════════════════════════════════════════════
#  5.  Main
# ══════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="Batch RCNN training — run a range of IDs on one GPU.",
        epilog="Example:  python train_rcnn_batch.py --start 0 --end 9 --gpu 0",
    )
    parser.add_argument("--start", type=int, required=True,
                        help="First run ID (inclusive)")
    parser.add_argument("--end", type=int, required=True,
                        help="Last run ID (inclusive)")
    parser.add_argument("--gpu", type=int, default=0,
                        help="CUDA device index (default: 0)")
    parser.add_argument("--num-epochs", type=int, default=1000,
                        help="Training epochs per run (default: 1000)")
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--print-every", type=int, default=50)
    parser.add_argument("--output-dir", type=str, default="rcnn_runs_32",
                        help="Directory to save results (default: rcnn_runs_32/)")
    parser.add_argument("--data-seed", type=int, default=42,
                        help="Seed for equilibrium & training data generation (default: 42)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # — Select GPU ————————————————————————————————————————————————
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # — Generate training data (deterministic, same on every GPU) —
    print("Generating equilibrium state and training data ...")
    set_seed(args.data_seed)

    data_generator = SyntheticDataGenerator(grid_size=(128, 128), device=device)
    equilibrium_state = data_generator.generate_equilibrium_state(
        equilibrium_precipitation=19.0, equilibrium_years=100,
    )

    n_sites = 4
    annual_totals = torch.linspace(16.5, 24.5, n_sites).tolist()
    peak_weeks = [26.0] * n_sites

    weekly_precipitation_profiles = [
        generate_weekly_precipitation(annual_total=at, peak_week=pw, amplitude_fraction=0.7)
        for at, pw in zip(annual_totals, peak_weeks)
    ]

    training_data = data_generator.generate_training_data(
        weekly_precipitation_profiles=weekly_precipitation_profiles,
        equilibrium_state=equilibrium_state,
        time_series_years=10,
        noise_level=EcologicalParameters.NOISE_LEVEL,
    )
    print(f"Data ready: {n_sites} sites × {len(training_data[0])} years "
          f"on {equilibrium_state.shape[-2]}×{equilibrium_state.shape[-1]} grid.\n")

    # — Run the requested range ——————————————————————————————————
    run_ids = list(range(args.start, args.end + 1))

    summary = []
    for rid in run_ids:
        # Distinct seed per run (deterministic, independent of data seed)
        seed = 1000 + rid

        result = train_single_run(
            run_id=rid,
            seed=seed,
            device=device,
            equilibrium_state=equilibrium_state,
            training_data=training_data,
            weekly_precipitation_profiles=weekly_precipitation_profiles,
            num_epochs=args.num_epochs,
            gradient_clip=args.gradient_clip,
            print_every=args.print_every,
        )

        # Save model checkpoint — filename contains run_id, so parallel
        # launches with non-overlapping ranges never collide.
        ckpt_path = os.path.join(args.output_dir, f"run_{rid:02d}.pt")
        torch.save({
            "run_id": result["run_id"],
            "seed": result["seed"],
            "model_state_dict": result["model_state_dict"],
            "loss_history": result["loss_history"],
            "num_epochs": result["num_epochs"],
            "final_loss": result["final_loss"],
            "elapsed_seconds": result["elapsed_seconds"],
        }, ckpt_path)
        print(f"  Saved → {ckpt_path}")

        summary.append({
            "run_id": result["run_id"],
            "seed": result["seed"],
            "num_epochs": result["num_epochs"],
            "final_loss": result["final_loss"],
            "elapsed_seconds": result["elapsed_seconds"],
        })

    # — Write summary JSON (one per range, so no collision) ———————
    summary_path = os.path.join(
        args.output_dir, f"summary_{args.start:02d}-{args.end:02d}.json",
    )
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary written to {summary_path}")

    # — Print final table —————————————————————————————————————————
    losses = [s["final_loss"] for s in summary if not math.isnan(s["final_loss"])]
    print(f"\n{'='*60}")
    print(f"  {len(summary)} runs completed  (IDs {args.start}–{args.end})")
    if losses:
        print(f"  Final loss — mean: {np.mean(losses):.6f}  "
              f"std: {np.std(losses):.6f}  "
              f"min: {np.min(losses):.6f}  max: {np.max(losses):.6f}")
    nan_runs = sum(1 for s in summary if math.isnan(s["final_loss"]))
    if nan_runs:
        print(f"  ⚠  {nan_runs} run(s) hit NaN")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
