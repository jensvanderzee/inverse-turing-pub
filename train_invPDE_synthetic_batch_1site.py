#!/usr/bin/env python3
"""
Batch Inverse PDE training script - run a range of training runs on a single GPU.

Derived from synthetic_invPDE_weekly notebook.

Usage
-----
# Launch 3 processes in parallel, each on a different GPU:
    ./launch.sh 0  9  0   # runs 0-9  on GPU 0
    ./launch.sh 10 19 1   # runs 10-19 on GPU 1
    ./launch.sh 20 29 2   # runs 20-29 on GPU 2

# Or call the Python script directly:
    python train_invPDE_batch.py --start 0 --end 29 --gpu 0
"""

import argparse
import json
import math
import os
import random
import time
from typing import Tuple, List, Optional

import numpy as np
import torch
import torch.nn as nn

# ===========================================================================
#  0.  Reproducibility
# ===========================================================================
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ===========================================================================
#  1.  Model Parameters & Generators
# ===========================================================================
class EcologicalParameters:
    # Diffusion coefficients
    SURFACE_WATER_DIFFUSION = 8.0
    SOIL_WATER_DIFFUSION = 1.0
    BIOMASS_DIFFUSION = 0.05

    # Loss rates
    EVAPORATION_RATE = 0.6
    SEEPAGE_RATE = 0.8
    MORTALITY_RATE = 0.6

    # Water cycle rates
    INFILTRATION_RATE = 0.2
    PLANT_UPTAKE_RATE = 0.35

    # Biological parameters
    BASE_PRECIPITATION = 0.675
    WATER_USE_EFFICIENCY = 0.35
    TIME_STEP = 1.0

    # Training parameters
    NOISE_LEVEL = 0.05

    # Weekly forcing defaults
    STEPS_PER_WEEK = 1
    NUM_WEEKS = 52


def generate_weekly_precipitation(annual_total: float, peak_week: float = 26.0,
                                  amplitude_fraction: float = 0.7,
                                  num_weeks: int = 52) -> List[float]:
    mean_rate = annual_total / 365.0  # mm/day
    raw = [mean_rate * (1.0 + amplitude_fraction * math.cos(2.0 * math.pi * (w - peak_week) / num_weeks))
           for w in range(num_weeks)]
    raw_total = sum(r * 7.0 for r in raw)
    scale = annual_total / raw_total if raw_total > 0 else 1.0
    return [r * scale for r in raw]


# ===========================================================================
#  2.  Core NCA Model
# ===========================================================================
class invRietkerk(nn.Module):
    def __init__(self, trainable: bool = False):
        super().__init__()
        self.base_time_step = EcologicalParameters.TIME_STEP
        self.trainable = trainable
        self._setup_spatial_operators()
        self._initialize_parameters()

    def _setup_spatial_operators(self):
        laplacian_kernel = torch.tensor([
            [0, 1, 0],
            [1, -4, 1],
            [0, 1, 0]
        ], dtype=torch.float32)

        for name in ['surface_water', 'soil_water', 'biomass']:
            conv = nn.Conv2d(1, 1, kernel_size=3, padding=1,
                           padding_mode="replicate", bias=False)
            conv.weight = nn.Parameter(laplacian_kernel[None, None, :], requires_grad=False)
            setattr(self, f"{name}_diffusion_op", conv)

    def _initialize_parameters(self):
        if self.trainable:
            self.infiltration_rate = nn.Parameter(torch.rand(1))
            self.seepage_rate = nn.Parameter(torch.rand(1))
            self.plant_uptake_rate = nn.Parameter(torch.rand(1))
            self.mortality_rate = nn.Parameter(torch.rand(1))
            self.evaporation_rate = nn.Parameter(torch.rand(1))
            self.water_use_efficiency = nn.Parameter(torch.rand(1))

            self.surface_water_diffusion_coeff = nn.Parameter(torch.rand(1))
            self.soil_water_diffusion_coeff = nn.Parameter(torch.rand(1))
            self.biomass_diffusion_coeff = nn.Parameter(torch.rand(1))
        else:
            self.infiltration_rate = EcologicalParameters.INFILTRATION_RATE
            self.evaporation_rate = EcologicalParameters.EVAPORATION_RATE
            self.seepage_rate = EcologicalParameters.SEEPAGE_RATE
            self.water_use_efficiency = EcologicalParameters.WATER_USE_EFFICIENCY
            self.plant_uptake_rate = EcologicalParameters.PLANT_UPTAKE_RATE
            
            self.mortality_rate = EcologicalParameters.MORTALITY_RATE
            
            self.surface_water_diffusion_coeff = EcologicalParameters.SURFACE_WATER_DIFFUSION
            self.soil_water_diffusion_coeff = EcologicalParameters.SOIL_WATER_DIFFUSION
            self.biomass_diffusion_coeff = EcologicalParameters.BIOMASS_DIFFUSION

    def forward(self, surface_water: torch.Tensor, soil_water: torch.Tensor,
                biomass: torch.Tensor, precipitation_rate: float = 0.0,
                time_step: float = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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

    def simulate_year_weekly(self, surface_water: torch.Tensor, soil_water: torch.Tensor,
                             biomass: torch.Tensor, weekly_precipitation: list,
                             steps_per_week: int = 3) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        current_surface_water = surface_water.clone()
        current_soil_water = soil_water.clone()
        current_biomass = biomass.clone()

        num_weeks = len(weekly_precipitation)
        total_steps_per_year = num_weeks * steps_per_week
        dt = 1.0 / total_steps_per_year

        for week in range(num_weeks):
            real_weekly_volume = weekly_precipitation[week] * 7.0
            volume_per_step = real_weekly_volume / steps_per_week
            adjusted_precip_rate = volume_per_step / dt

            for step in range(steps_per_week):
                current_surface_water, current_soil_water, current_biomass = self.forward(
                    current_surface_water, current_soil_water, current_biomass,
                    precipitation_rate=adjusted_precip_rate,
                    time_step=dt
                )

        return current_surface_water, current_soil_water, current_biomass


# ===========================================================================
#  3.  Data Generator
# ===========================================================================
class SyntheticDataGenerator:
    def __init__(self, grid_size: Tuple[int, int] = (128, 128), device: Optional[torch.device] = None):
        self.height, self.width = grid_size
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.ground_truth_model = invRietkerk(trainable=False).to(self.device)

    def generate_equilibrium_state(self, equilibrium_precipitation: float = 43.0,
                                   equilibrium_years: int = 50,
                                   steps_per_week: int = 3) -> torch.Tensor:
        surface_water = torch.rand(1, 1, self.height, self.width, device=self.device)*10
        soil_water = torch.rand(1, 1, self.height, self.width, device=self.device)*10
        biomass = torch.rand(1, 1, self.height, self.width, device=self.device)*10

        uniform_weekly = generate_weekly_precipitation(
            equilibrium_precipitation, peak_week=26.0, amplitude_fraction=0.0)

        for _ in range(equilibrium_years):
            surface_water, soil_water, biomass = self.ground_truth_model.simulate_year_weekly(
                surface_water, soil_water, biomass,
                weekly_precipitation=uniform_weekly,
                steps_per_week=steps_per_week
            )
        return biomass.clone()

    def generate_training_data(self, weekly_precipitation_profiles: List[List[float]],
                               equilibrium_state: torch.Tensor,
                               time_series_years: int = 10,
                               noise_level: float = 0.02,
                               steps_per_week: int = 3) -> List[List[torch.Tensor]]:
        training_data = []
        for site_idx, weekly_precip in enumerate(weekly_precipitation_profiles):
            surface_water = torch.zeros(1, 1, self.height, self.width, device=self.device)
            soil_water = torch.zeros(1, 1, self.height, self.width, device=self.device)
            biomass = equilibrium_state.clone()

            site_time_series = []
            for year in range(time_series_years):
                surface_water, soil_water, biomass = self.ground_truth_model.simulate_year_weekly(
                    surface_water, soil_water, biomass,
                    weekly_precipitation=weekly_precip,
                    steps_per_week=steps_per_week
                )
                noise = torch.randn_like(biomass) * noise_level
                site_time_series.append(biomass + noise)

            training_data.append(site_time_series)
        return training_data


# ===========================================================================
#  4.  Training Routine
# ===========================================================================
def train_model_adam(training_data: List[List[torch.Tensor]],
                    equilibrium_state: torch.Tensor,
                    weekly_precipitation_profiles: List[List[float]],
                    num_epochs: int = 500,
                    learning_rate: float = 0.001,
                    device: torch.device = None,
                    seed: int = None,
                    save_interval: int = 10,
                    steps_per_week: int = 3) -> Tuple[List[float], invRietkerk, dict, List[dict]]:

    if seed is not None:
        set_seed(seed)

    model = invRietkerk(trainable=True).to(device)
    loss_function = nn.MSELoss()

    initial_params = {name: param.item() for name, param in model.named_parameters() if param.data.shape == torch.Size([1])}
    parameter_history = []

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.95, 0.99), eps=1e-8)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9999)
    loss_history = []

    def compute_loss():
        total_loss = 0
        for site_idx, weekly_precip in enumerate(weekly_precipitation_profiles):
            pred_surface_water = torch.zeros(1, 1, 128, 128, device=device)
            pred_soil_water = torch.zeros(1, 1, 128, 128, device=device)
            pred_biomass = equilibrium_state.clone()

            prev_pred_biomass = pred_biomass.clone()
            prev_target_biomass = pred_biomass.clone()

            for year_idx in range(len(training_data[site_idx])):
                pred_surface_water, pred_soil_water, pred_biomass = model.simulate_year_weekly(
                    pred_surface_water, pred_soil_water, pred_biomass,
                    weekly_precipitation=weekly_precip,
                    steps_per_week=steps_per_week
                )
                target_biomass = training_data[site_idx][year_idx]
                biomass_loss = loss_function((pred_biomass - prev_pred_biomass), (target_biomass - prev_target_biomass))
                prev_pred_biomass = pred_biomass
                prev_target_biomass = target_biomass
                total_loss += biomass_loss
        return total_loss

    def capture_parameters(epoch):
        params = {'epoch': epoch}
        for name, param in model.named_parameters():
            if param.data.shape == torch.Size([1]):
                params[name] = param.item()
        return params

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        loss = compute_loss()
        loss.backward()
        optimizer.step()
        scheduler.step()

        with torch.no_grad():
            for param in model.parameters():
                if param.data.shape == torch.Size([1]):
                    param.data.clamp_(0.0001, 10000)

        loss_value = loss.item()
        loss_history.append(loss_value)

        if epoch % save_interval == 0:
            parameter_history.append(capture_parameters(epoch))

        if math.isnan(loss_value):
            print(f"NaN loss encountered at epoch {epoch}", flush=True)
            break

        if epoch % 10 == 0 or epoch == num_epochs - 1:
            param_str = "\n  ".join(f"{name}: {param.item():.4f}"
                                  for name, param in model.named_parameters()
                                  if param.data.shape == torch.Size([1]))
            print(f"Epoch {epoch:4d}, Loss: {loss_value:.6f}  |  {param_str}", flush=True)

    if (num_epochs - 1) % save_interval != 0:
        parameter_history.append(capture_parameters(num_epochs - 1))

    return loss_history, model, initial_params, parameter_history


# ===========================================================================
#  5.  Main Batch Loop
# ===========================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=0, help="Start run ID (inclusive)")
    parser.add_argument("--end", type=int, default=29, help="End run ID (inclusive)")
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID to use")
    parser.add_argument("--output_dir", type=str, default="synthetic_results_1site", help="Output directory")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}", flush=True)

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(f"{args.output_dir}/models", exist_ok=True)
    os.makedirs(f"{args.output_dir}/results", exist_ok=True)

    print("Generating training data...", flush=True)
    set_seed(42)  # Base seed so all runs share the same ground-truth

    data_generator = SyntheticDataGenerator(grid_size=(128, 128), device=device)
    equilibrium_state = data_generator.generate_equilibrium_state(
        equilibrium_precipitation=19.0, equilibrium_years=100
    )

    n_sites = 1
    annual_totals = torch.linspace(16.5, 24.5, n_sites).tolist()
    peak_weeks = [26.0, 26.0, 26.0, 26.0]

    weekly_precipitation_profiles = [
        generate_weekly_precipitation(annual_total=at, peak_week=pw, amplitude_fraction=0.7)
        for at, pw in zip(annual_totals, peak_weeks)
    ]

    training_data = data_generator.generate_training_data(
        weekly_precipitation_profiles,
        equilibrium_state,
        time_series_years=10,
        noise_level=0.05,
        steps_per_week=1
    )

    summary = []

    for run_id in range(args.start, args.end + 1):
        print(f"\n{'='*60}", flush=True)
        print(f" Starting run ID {run_id}", flush=True)
        print(f"{'='*60}", flush=True)

        run_seed = 42 + run_id
        start_time = time.time()

        loss_history, model, initial_params, parameter_history = train_model_adam(
            training_data=training_data,
            equilibrium_state=equilibrium_state,
            weekly_precipitation_profiles=weekly_precipitation_profiles,
            num_epochs=10000,
            learning_rate=0.1,
            device=device,
            seed=run_seed,
            save_interval=10,
            steps_per_week=1
        )

        elapsed = time.time() - start_time
        final_loss = loss_history[-1] if len(loss_history) > 0 else float('nan')

        # Save model
        model_path = os.path.join(args.output_dir, "models", f"invPDE_run_{run_id:02d}.pt")
        torch.save(model.state_dict(), model_path)

        # Save run JSON
        result_dict = {
            "run_id": run_id,
            "seed": run_seed,
            "initial_params": initial_params,
            "parameter_history": parameter_history,
            "loss_history": loss_history,
            "num_epochs": len(loss_history),
            "final_loss": final_loss,
            "elapsed_seconds": elapsed
        }

        result_path = os.path.join(args.output_dir, "results", f"result_{run_id:02d}.json")
        with open(result_path, "w") as f:
            json.dump(result_dict, f, indent=2)

        print(f"  Saved -> {result_path}", flush=True)

        summary.append({
            "run_id": run_id,
            "seed": run_seed,
            "num_epochs": len(loss_history),
            "final_loss": final_loss,
            "elapsed_seconds": elapsed
        })

    # Write summary block
    summary_path = os.path.join(args.output_dir, f"summary_{args.start:02d}-{args.end:02d}.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSummary written to {summary_path}", flush=True)

    losses = [s["final_loss"] for s in summary if not math.isnan(s["final_loss"])]
    print(f"\n{'='*60}", flush=True)
    print(f"  {len(summary)} runs completed (IDs {args.start}-{args.end})", flush=True)
    if losses:
        print(f"  Final loss - mean: {np.mean(losses):.6f} std: {np.std(losses):.6f}", flush=True)


if __name__ == "__main__":
    main()