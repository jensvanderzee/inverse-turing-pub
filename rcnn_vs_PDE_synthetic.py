"""
Compare PDE parameter learning vs RCNN on extrapolation tasks
"""
#%%
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import pickle
import json
import os
from typing import Tuple, List, Optional
import seaborn as sns

# Set style for better plots
plt.style.use('default')
sns.set_palette("husl")

# Model classes (same as in training scripts)
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
    NOISE_LEVEL = 0.05
    SAMPLE_INTERVAL = 50

class invRietkerk(nn.Module):
    def __init__(self, trainable: bool = False):
        super().__init__()
        self.time_step = EcologicalParameters.TIME_STEP
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
            self.precipitation = EcologicalParameters.BASE_PRECIPITATION
            self.infiltration_rate = nn.Parameter(torch.rand(1))
            self.seepage_rate = nn.Parameter(torch.rand(1))
            self.plant_uptake_rate = nn.Parameter(torch.rand(1))
            self.mortality_rate = nn.Parameter(torch.rand(1))
            self.evaporation_rate = nn.Parameter(torch.rand(1))
            self.water_use_efficiency = nn.Parameter(torch.rand(1))
            self.surface_water_diffusion_coeff = nn.Parameter(torch.rand(1))
            self.soil_water_diffusion_coeff = nn.Parameter(torch.rand(1))
            self.biomass_diffusion_coeff = nn.Parameter(torch.rand(1))
            self.growth_factor_eta = EcologicalParameters.GROWTH_FACTOR_ETA
            self.growth_exponent_q = EcologicalParameters.GROWTH_EXPONENT_Q
        else:
            # Ground truth parameters for comparison
            self.precipitation = EcologicalParameters.BASE_PRECIPITATION
            self.infiltration_rate = EcologicalParameters.INFILTRATION_RATE
            self.evaporation_rate = EcologicalParameters.EVAPORATION_RATE
            self.seepage_rate = EcologicalParameters.SEEPAGE_RATE
            self.plant_uptake_rate = EcologicalParameters.PLANT_UPTAKE_RATE
            self.growth_factor_eta = EcologicalParameters.GROWTH_FACTOR_ETA
            self.mortality_rate = EcologicalParameters.MORTALITY_RATE
            self.water_use_efficiency = EcologicalParameters.WATER_USE_EFFICIENCY
            self.growth_exponent_q = EcologicalParameters.GROWTH_EXPONENT_Q
            self.surface_water_diffusion_coeff = EcologicalParameters.SURFACE_WATER_DIFFUSION
            self.soil_water_diffusion_coeff = EcologicalParameters.SOIL_WATER_DIFFUSION
            self.biomass_diffusion_coeff = EcologicalParameters.BIOMASS_DIFFUSION

    def forward(self, surface_water: torch.Tensor, soil_water: torch.Tensor,
                biomass: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        growth_term = (1 + self.growth_factor_eta * biomass ** self.growth_exponent_q)

        surface_water = surface_water + (
            self.surface_water_diffusion_coeff * self.surface_water_diffusion_op(surface_water)
            - self.evaporation_rate * surface_water
            + self.precipitation
            - self.infiltration_rate * surface_water * biomass
        ) * self.time_step

        soil_water = soil_water + (
            self.soil_water_diffusion_coeff * self.soil_water_diffusion_op(soil_water)
            - self.seepage_rate * soil_water
            + self.infiltration_rate * surface_water * biomass
            - self.plant_uptake_rate * soil_water * biomass * growth_term
        ) * self.time_step

        biomass = biomass + (
            self.biomass_diffusion_coeff * self.biomass_diffusion_op(biomass)
            - self.mortality_rate * biomass
            + self.water_use_efficiency * self.plant_uptake_rate * soil_water * biomass * growth_term
        ) * self.time_step

        return surface_water, soil_water, biomass

class RecurrentConvNet(nn.Module):
    def __init__(self, hidden_channels: int = 32, num_conv_layers: int = 3,
                 kernel_size: int = 3):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.num_conv_layers = num_conv_layers

        self.input_conv = nn.Conv2d(2 + hidden_channels, hidden_channels, kernel_size=1)

        self.conv_blocks = nn.ModuleList()
        for i in range(num_conv_layers):
            self.conv_blocks.append(
                nn.Sequential(
                    nn.Conv2d(hidden_channels, hidden_channels,
                             kernel_size=kernel_size,
                             padding=kernel_size//2,
                             padding_mode='replicate'),
                    nn.GroupNorm(8, hidden_channels),
                    nn.ReLU(inplace=True)
                )
            )

        self.temporal_processing = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels * 2, kernel_size=3, padding=1),
            nn.GroupNorm(16, hidden_channels * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels * 2, hidden_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden_channels),
            nn.ReLU(inplace=True)
        )

        self.hidden_update = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1),
            nn.Tanh()
        )

        self.output_conv = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels // 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels // 2, 1, kernel_size=1),
            nn.Sigmoid()
        )

        self.output_scale = nn.Parameter(torch.tensor(2.0))
        self._initialize_weights()

    def _initialize_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                if "hidden_update" in name or "input_conv" in name:
                    nn.init.orthogonal_(m.weight, gain=1.0)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                else:
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, biomass: torch.Tensor, precipitation: float,
                hidden_state: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, _, height, width = biomass.shape
        device = biomass.device

        if hidden_state is None:
            hidden_state = torch.zeros(batch_size, self.hidden_channels, height, width, device=device)

        precip_channel = torch.full_like(biomass, precipitation)
        x = torch.cat([biomass, precip_channel, hidden_state], dim=1)
        x = self.input_conv(x)

        for conv_block in self.conv_blocks:
            residual = x
            x = conv_block(x)
            x = x + residual

        x = self.temporal_processing(x) + x
        new_hidden_state = self.hidden_update(x) + hidden_state * 0.5
        next_year_biomass = self.output_conv(x) * self.output_scale

        return next_year_biomass, new_hidden_state

class SyntheticDataGenerator:
    def __init__(self, grid_size: Tuple[int, int] = (150, 150), device: Optional[torch.device] = None):
        self.height, self.width = grid_size
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.ground_truth_model = invRietkerk(trainable=False).to(self.device)

    def generate_equilibrium_state(self, equilibrium_precipitation: float = 43.0,
                                 equilibrium_steps: int = 2500) -> torch.Tensor:
        sample_interval = EcologicalParameters.SAMPLE_INTERVAL

        surface_water = torch.rand(1, 1, self.height, self.width, device=self.device)
        soil_water = torch.rand(1, 1, self.height, self.width, device=self.device)
        biomass = torch.rand(1, 1, self.height, self.width, device=self.device)

        self.ground_truth_model.precipitation = 0

        for step in range(equilibrium_steps):
            if step % sample_interval == 0:
                self.ground_truth_model.precipitation = equilibrium_precipitation

            surface_water, soil_water, biomass = self.ground_truth_model.forward(
                surface_water, soil_water, biomass
            )
            self.ground_truth_model.precipitation = 0

        return biomass.clone()

def load_best_models(pde_dir: str, rcnn_dir: str, device: torch.device):
    """Load the best models from both approaches"""
    
    # Load PDE model results
    pde_summary_path = os.path.join(pde_dir, "results", "training_summary.json")
    if os.path.exists(pde_summary_path):
        with open(pde_summary_path, 'r') as f:
            pde_summary = json.load(f)
        best_pde_id = pde_summary['best_model_id']
        best_pde_loss = pde_summary['best_loss']
    else:
        # Fallback: scan all models
        best_pde_id = 0
        best_pde_loss = float('inf')
        
    print(f"Best PDE model: Model {best_pde_id} with loss {best_pde_loss:.6f}")
    
    # Load best PDE model
    pde_model = invRietkerk(trainable=True).to(device)
    pde_model_path = os.path.join(pde_dir, "models", f"model_{best_pde_id:02d}.pth")
    pde_model.load_state_dict(torch.load(pde_model_path, map_location=device))
    pde_model.eval()
    
    # Load RCNN model results
    rcnn_results_path = os.path.join(rcnn_dir, "results", "all_results.pkl")
    with open(rcnn_results_path, 'rb') as f:
        rcnn_results = pickle.load(f)
    
    # Find best RCNN model
    successful_rcnn = [r for r in rcnn_results if not np.isnan(r['final_loss'])]
    best_rcnn = min(successful_rcnn, key=lambda x: x['final_loss'])
    best_rcnn_id = best_rcnn['model_id']
    best_rcnn_loss = best_rcnn['final_loss']
    best_rcnn_config = best_rcnn['model_config']
    
    print(f"Best RCNN model: Model {best_rcnn_id} with loss {best_rcnn_loss:.6f}")
    print(f"RCNN config: {best_rcnn_config}")
    
    # Load best RCNN model
    rcnn_model = RecurrentConvNet(**best_rcnn_config).to(device)
    rcnn_model_path = os.path.join(rcnn_dir, "models", f"model_{best_rcnn_id:02d}.pth")
    rcnn_checkpoint = torch.load(rcnn_model_path, map_location=device)
    rcnn_model.load_state_dict(rcnn_checkpoint['model_state_dict'])
    rcnn_model.eval()
    
    return {
        'pde_model': pde_model,
        'rcnn_model': rcnn_model,
        'pde_info': {'id': best_pde_id, 'loss': best_pde_loss},
        'rcnn_info': {'id': best_rcnn_id, 'loss': best_rcnn_loss, 'config': best_rcnn_config}
    }

def run_pde_simulation(model, initial_biomass, precipitation_values, time_steps, device):
    """Run PDE model simulation"""
    n_sites = len(precipitation_values)
    batch_size, _, height, width = initial_biomass.shape
    sample_interval = EcologicalParameters.SAMPLE_INTERVAL
    
    # Initialize states for all sites
    all_predictions = []
    
    for site_idx, precip_value in enumerate(precipitation_values):
        surface_water = torch.zeros(batch_size, 1, height, width, device=device)
        soil_water = torch.zeros(batch_size, 1, height, width, device=device)
        biomass = initial_biomass.clone()
        
        site_predictions = [biomass.clone()]
        
        for step in range(time_steps * sample_interval):
            model.precipitation = 0
            if step % sample_interval == 0:
                model.precipitation = precip_value
                
            surface_water, soil_water, biomass = model.forward(surface_water, soil_water, biomass)
            
            if step % sample_interval == 0 and step > 0:
                site_predictions.append(biomass.clone())
        
        all_predictions.append(site_predictions)
    
    return all_predictions

def run_rcnn_simulation(model, initial_biomass, precipitation_values, time_steps, device):
    """Run RCNN model simulation"""
    all_predictions = []
    
    for site_idx, precip_value in enumerate(precipitation_values):
        biomass = initial_biomass.clone()
        hidden_state = None
        
        site_predictions = [biomass.clone()]
        
        for step in range(time_steps):
            biomass, hidden_state = model(biomass, precip_value.item(), hidden_state)
            site_predictions.append(biomass.clone())
            
            # Detach to prevent gradient accumulation
            biomass = biomass.detach()
            hidden_state = hidden_state.detach()
        
        all_predictions.append(site_predictions)
    
    return all_predictions

def generate_ground_truth(initial_biomass, precipitation_values, time_steps, device):
    """Generate ground truth using the original PDE model"""
    ground_truth_model = invRietkerk(trainable=False).to(device)
    return run_pde_simulation(ground_truth_model, initial_biomass, precipitation_values, time_steps, device)

def compute_extrapolation_metrics(predictions, ground_truth):
    """Compute various metrics for extrapolation performance"""
    metrics = {
        'mse': [],
        'mae': [],
        'relative_error': [],
        'spatial_correlation': []
    }
    
    for site_idx, (pred_seq, gt_seq) in enumerate(zip(predictions, ground_truth)):
        site_metrics = {'mse': [], 'mae': [], 'relative_error': [], 'spatial_correlation': []}
        
        for t, (pred, gt) in enumerate(zip(pred_seq, gt_seq)):
            # Convert to numpy for easier computation
            pred_np = pred.detach().cpu().numpy().flatten()
            gt_np = gt.detach().cpu().numpy().flatten()
            
            # MSE
            mse = np.mean((pred_np - gt_np) ** 2)
            site_metrics['mse'].append(mse)
            
            # MAE
            mae = np.mean(np.abs(pred_np - gt_np))
            site_metrics['mae'].append(mae)
            
            # Relative error
            rel_error = np.mean(np.abs(pred_np - gt_np) / (gt_np + 1e-8))
            site_metrics['relative_error'].append(rel_error)
            
            # Spatial correlation
            correlation = np.corrcoef(pred_np, gt_np)[0, 1]
            if not np.isnan(correlation):
                site_metrics['spatial_correlation'].append(correlation)
            else:
                site_metrics['spatial_correlation'].append(0.0)
        
        for key in metrics:
            metrics[key].append(site_metrics[key])
    
    return metrics

def run_extrapolation_experiments(models, device):
    """Run comprehensive extrapolation experiments"""
    
    # Generate initial equilibrium state
    data_generator = SyntheticDataGenerator(grid_size=(150, 150), device=device)
    equilibrium_state = data_generator.generate_equilibrium_state(
        equilibrium_precipitation=43.0, equilibrium_steps=2500
    )
    
    experiments = {}
    
    # Experiment 1: Temporal extrapolation (longer time series)
    print("Running temporal extrapolation experiment...")
    train_precip = torch.linspace(43, 53, 4)  # Same as training
    extended_time_steps = 20  # Double the training length (was 10)
    
    # Generate ground truth
    gt_temporal = generate_ground_truth(equilibrium_state, train_precip, extended_time_steps, device)
    
    # Run models
    pde_temporal = run_pde_simulation(models['pde_model'], equilibrium_state, train_precip, extended_time_steps, device)
    rcnn_temporal = run_rcnn_simulation(models['rcnn_model'], equilibrium_state, train_precip, extended_time_steps, device)
    
    experiments['temporal'] = {
        'ground_truth': gt_temporal,
        'pde_predictions': pde_temporal,
        'rcnn_predictions': rcnn_temporal,
        'precipitation_values': train_precip,
        'description': 'Extended time series (double training length)'
    }
    
    # Experiment 2: Precipitation extrapolation (unseen precipitation values)
    print("Running precipitation extrapolation experiment...")
    # Test on precipitation values outside the training range (43-53)
    extrap_precip = torch.tensor([35.0, 40.0, 58.0, 65.0])  # Outside training range
    normal_time_steps = 100  # Same as training length
    
    gt_precip = generate_ground_truth(equilibrium_state, extrap_precip, normal_time_steps, device)
    pde_precip = run_pde_simulation(models['pde_model'], equilibrium_state, extrap_precip, normal_time_steps, device)
    rcnn_precip = run_rcnn_simulation(models['rcnn_model'], equilibrium_state, extrap_precip, normal_time_steps, device)
    
    experiments['precipitation'] = {
        'ground_truth': gt_precip,
        'pde_predictions': pde_precip,
        'rcnn_predictions': rcnn_precip,
        'precipitation_values': extrap_precip,
        'description': 'Unseen precipitation values (outside training range 43-53)'
    }
    
    # Experiment 3: Combined extrapolation (both time and precipitation)
    print("Running combined extrapolation experiment...")
    combined_precip = torch.tensor([38.0, 60.0])  # Extreme values
    combined_time_steps = 100  # Longer time series
    
    gt_combined = generate_ground_truth(equilibrium_state, combined_precip, combined_time_steps, device)
    pde_combined = run_pde_simulation(models['pde_model'], equilibrium_state, combined_precip, combined_time_steps, device)
    rcnn_combined = run_rcnn_simulation(models['rcnn_model'], equilibrium_state, combined_precip, combined_time_steps, device)
    
    experiments['combined'] = {
        'ground_truth': gt_combined,
        'pde_predictions': pde_combined,
        'rcnn_predictions': rcnn_combined,
        'precipitation_values': combined_precip,
        'description': 'Combined temporal and precipitation extrapolation'
    }
    
    # Experiment 4: Precipitation range analysis (25-75 range, 100 years)
    print("Running precipitation range analysis (25-75, 100 years)...")
    precip_range = torch.linspace(25, 75, 21)  # 21 points from 25 to 75
    range_time_steps = 100  # 100 years simulation
    
    # Store only final results to save memory
    range_results = {
        'precipitation_values': precip_range.clone(),
        'gt_final_biomass': [],
        'pde_final_biomass': [],
        'rcnn_final_biomass': [],
        'pde_correlations': [],
        'rcnn_correlations': [],
        'description': 'Precipitation range analysis (25-75 mm, 100 years simulation)'
    }
    
    print(f"Processing {len(precip_range)} precipitation values one by one...")
    
    for i, precip_val in enumerate(precip_range):
        print(f"Processing precipitation {precip_val:.1f} mm ({i+1}/{len(precip_range)})")
        
        # Process one precipitation value at a time
        single_precip = torch.tensor([precip_val])
        
        try:
            # Run simulations for this single precipitation value
            gt_single = generate_ground_truth(equilibrium_state, single_precip, range_time_steps, device)
            pde_single = run_pde_simulation(models['pde_model'], equilibrium_state, single_precip, range_time_steps, device)
            rcnn_single = run_rcnn_simulation(models['rcnn_model'], equilibrium_state, single_precip, range_time_steps, device)
            
            # Extract only final values and correlations
            gt_final = torch.mean(gt_single[0][-1][0, 0]).item()
            pde_final = torch.mean(pde_single[0][-1][0, 0]).item()
            rcnn_final = torch.mean(rcnn_single[0][-1][0, 0]).item()
            
            # Calculate spatial correlations with ground truth
            gt_flat = gt_single[0][-1][0, 0].cpu().numpy().flatten()
            pde_flat = pde_single[0][-1][0, 0].detach().cpu().numpy().flatten()
            rcnn_flat = rcnn_single[0][-1][0, 0].detach().cpu().numpy().flatten()
            
            pde_corr = np.corrcoef(gt_flat, pde_flat)[0, 1]
            rcnn_corr = np.corrcoef(gt_flat, rcnn_flat)[0, 1]
            
            # Store results
            range_results['gt_final_biomass'].append(gt_final)
            range_results['pde_final_biomass'].append(pde_final)
            range_results['rcnn_final_biomass'].append(rcnn_final)
            range_results['pde_correlations'].append(pde_corr if not np.isnan(pde_corr) else 0.0)
            range_results['rcnn_correlations'].append(rcnn_corr if not np.isnan(rcnn_corr) else 0.0)
            
            # Explicitly delete large objects and force garbage collection
            del gt_single, pde_single, rcnn_single
            del gt_flat, pde_flat, rcnn_flat
            import gc
            gc.collect()
            
            # Print memory usage
            import psutil
            memory_percent = psutil.virtual_memory().percent
            print(f"  Memory usage: {memory_percent:.1f}%")
            
            # Safety check
            if memory_percent > 85:
                print(f"WARNING: High memory usage ({memory_percent:.1f}%), stopping early")
                break
                
        except Exception as e:
            print(f"Error processing precipitation {precip_val:.1f}: {e}")
            range_results['gt_final_biomass'].append(np.nan)
            range_results['pde_final_biomass'].append(np.nan)
            range_results['rcnn_final_biomass'].append(np.nan)
            range_results['pde_correlations'].append(np.nan)
            range_results['rcnn_correlations'].append(np.nan)
    
    experiments['precipitation_range'] = range_results
    
    return experiments

def visualize_results(experiments, models, save_dir):
    """Create comprehensive visualizations of extrapolation results"""
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Color scheme
    colors = {'PDE': '#2E8B57', 'RCNN': '#CD853F', 'Ground Truth': '#1C1C1C'}
    
    for exp_name, exp_data in experiments.items():
        print(f"Visualizing {exp_name} experiment...")
        
        # Special handling for precipitation_range experiment
        if exp_name == 'precipitation_range':
            continue  # Skip this - will be handled by the new visualization functions
        
        # For all other experiments
        pde_metrics = compute_extrapolation_metrics(exp_data['pde_predictions'], exp_data['ground_truth'])
        rcnn_metrics = compute_extrapolation_metrics(exp_data['rcnn_predictions'], exp_data['ground_truth'])
        
        # Create figure with multiple subplots
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Spatial snapshots at different time points
        n_sites = len(exp_data['precipitation_values'])
        time_points = [0, len(exp_data['ground_truth'][0])//3, 2*len(exp_data['ground_truth'][0])//3, -1]
        
        for i, t_idx in enumerate(time_points):
            for j, site_idx in enumerate(range(min(2, n_sites))):
                ax = plt.subplot(5, 4, i*2 + j + 1)
                
                # Ground truth
                gt_img = exp_data['ground_truth'][site_idx][t_idx][0, 0].cpu().numpy()
                im = ax.imshow(gt_img, cmap='viridis', vmin=0, vmax=2)
                ax.set_title(f'GT: Site {site_idx+1}, t={t_idx if t_idx >= 0 else "final"}\nPrecip: {exp_data["precipitation_values"][site_idx]:.1f}')
                ax.axis('off')
                plt.colorbar(im, ax=ax, shrink=0.6)
        
        # 2. Time series of spatial averages
        ax_ts = plt.subplot(5, 2, 5)
        for site_idx in range(n_sites):
            gt_means = [torch.mean(frame[0, 0]).item() for frame in exp_data['ground_truth'][site_idx]]
            pde_means = [torch.mean(frame[0, 0]).item() for frame in exp_data['pde_predictions'][site_idx]]
            rcnn_means = [torch.mean(frame[0, 0]).item() for frame in exp_data['rcnn_predictions'][site_idx]]
            min_length = min(len(gt_means), len(pde_means), len(rcnn_means))
            
            gt_means = gt_means[:min_length]
            pde_means = pde_means[:min_length]
            rcnn_means = rcnn_means[:min_length]
            time_axis = range(len(gt_means))
            
            plt.plot(time_axis, gt_means, 'k-', linewidth=2, alpha=0.8, 
                    label=f'GT Site {site_idx+1}' if site_idx == 0 else None)
            plt.plot(time_axis, pde_means, '--', color=colors['PDE'], linewidth=2, alpha=0.8,
                    label=f'PDE Site {site_idx+1}' if site_idx == 0 else None)
            plt.plot(time_axis, rcnn_means, ':', color=colors['RCNN'], linewidth=2, alpha=0.8,
                    label=f'RCNN Site {site_idx+1}' if site_idx == 0 else None)
        
        plt.xlabel('Time Step (Years)')
        plt.ylabel('Mean Biomass')
        plt.title(f'Temporal Evolution - {exp_name.title()} Extrapolation')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 3. MSE over time
        ax_mse = plt.subplot(5, 2, 6)
        for site_idx in range(n_sites):
            time_axis = range(len(pde_metrics['mse'][site_idx]))
            plt.plot(time_axis, pde_metrics['mse'][site_idx], '--', color=colors['PDE'], 
                    label=f'PDE Site {site_idx+1}' if site_idx == 0 else None)
            plt.plot(time_axis, rcnn_metrics['mse'][site_idx], ':', color=colors['RCNN'],
                    label=f'RCNN Site {site_idx+1}' if site_idx == 0 else None)
        
        plt.xlabel('Time Step')
        plt.ylabel('MSE')
        plt.title('MSE Evolution')
        plt.yscale('log')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 4. Correlation over time
        ax_corr = plt.subplot(5, 2, 7)
        for site_idx in range(n_sites):
            time_axis = range(len(pde_metrics['spatial_correlation'][site_idx]))
            plt.plot(time_axis, pde_metrics['spatial_correlation'][site_idx], '--', color=colors['PDE'],
                    label=f'PDE Site {site_idx+1}' if site_idx == 0 else None)
            plt.plot(time_axis, rcnn_metrics['spatial_correlation'][site_idx], ':', color=colors['RCNN'],
                    label=f'RCNN Site {site_idx+1}' if site_idx == 0 else None)
        
        plt.xlabel('Time Step')
        plt.ylabel('Spatial Correlation')
        plt.title('Spatial Pattern Preservation')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 5. Summary metrics
        ax_summary = plt.subplot(5, 2, 8)
        
        # Compute final metrics (average over last 3 time steps)
        final_steps = 3
        pde_final_mse = np.mean([np.mean(site_mse[-final_steps:]) for site_mse in pde_metrics['mse']])
        rcnn_final_mse = np.mean([np.mean(site_mse[-final_steps:]) for site_mse in rcnn_metrics['mse']])
        
        pde_final_corr = np.mean([np.mean(site_corr[-final_steps:]) for site_corr in pde_metrics['spatial_correlation']])
        rcnn_final_corr = np.mean([np.mean(site_corr[-final_steps:]) for site_corr in rcnn_metrics['spatial_correlation']])
        
        metrics_names = ['Final MSE', 'Final Correlation']
        pde_values = [pde_final_mse, pde_final_corr]
        rcnn_values = [rcnn_final_mse, rcnn_final_corr]
        
        x = np.arange(len(metrics_names))
        width = 0.35
        
        bars1 = ax_summary.bar(x - width/2, [pde_values[0], pde_values[1]], width, 
                              label='PDE', color=colors['PDE'], alpha=0.7)
        bars2 = ax_summary.bar(x + width/2, [rcnn_values[0], rcnn_values[1]], width,
                              label='RCNN', color=colors['RCNN'], alpha=0.7)
        
        ax_summary.set_ylabel('Value')
        ax_summary.set_title('Summary Metrics')
        ax_summary.set_xticks(x)
        ax_summary.set_xticklabels(metrics_names)
        ax_summary.legend()
        
        # Add value labels on bars
        for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
            height1 = bar1.get_height()
            height2 = bar2.get_height()
            ax_summary.text(bar1.get_x() + bar1.get_width()/2., height1,
                           f'{height1:.4f}', ha='center', va='bottom', fontsize=10)
            ax_summary.text(bar2.get_x() + bar2.get_width()/2., height2,
                           f'{height2:.4f}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'{exp_name}_extrapolation_analysis.png'), 
                   dpi=150, bbox_inches='tight')
        plt.show()
        
        # Print summary
        print(f"\n{exp_name.upper()} EXTRAPOLATION RESULTS:")
        print(f"Description: {exp_data['description']}")
        print(f"PDE Model - Final MSE: {pde_final_mse:.6f}, Final Correlation: {pde_final_corr:.4f}")
        print(f"RCNN Model - Final MSE: {rcnn_final_mse:.6f}, Final Correlation: {rcnn_final_corr:.4f}")
        
        if pde_final_mse < rcnn_final_mse:
            print(f"✓ PDE model performs better (MSE {pde_final_mse/rcnn_final_mse:.2f}x lower)")
        else:
            print(f"✓ RCNN model performs better (MSE {rcnn_final_mse/pde_final_mse:.2f}x lower)")

def create_precipitation_biomass_relationship(experiments, save_dir):
    """Create line plot showing precipitation vs mean biomass after 100 years"""
    
    if 'precipitation_range' not in experiments:
        print("Precipitation range experiment not found, skipping relationship plot")
        return
    
    range_data = experiments['precipitation_range']
    
    # Extract data
    precip_values = np.array(range_data['precipitation_values'])
    gt_biomass = np.array(range_data['gt_final_biomass'])
    pde_biomass = np.array(range_data['pde_final_biomass'])
    rcnn_biomass = np.array(range_data['rcnn_final_biomass'])
    
    # Remove any NaN values
    valid_idx = ~(np.isnan(gt_biomass) | np.isnan(pde_biomass) | np.isnan(rcnn_biomass))
    precip_values = precip_values[valid_idx]
    gt_biomass = gt_biomass[valid_idx]
    pde_biomass = pde_biomass[valid_idx]
    rcnn_biomass = rcnn_biomass[valid_idx]
    
    # Create the plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Colors
    colors = {'Ground Truth': '#1C1C1C', 'PDE': '#2E8B57', 'RCNN': '#CD853F'}
    
    # Left plot: Precipitation-Biomass Relationship
    ax1.plot(precip_values, gt_biomass, 'o-', color=colors['Ground Truth'], 
             linewidth=2.5, markersize=8, label='Ground Truth', alpha=0.9)
    ax1.plot(precip_values, pde_biomass, 's--', color=colors['PDE'], 
             linewidth=2, markersize=7, label='PDE Model', alpha=0.8)
    ax1.plot(precip_values, rcnn_biomass, '^:', color=colors['RCNN'], 
             linewidth=2, markersize=7, label='RCNN Model', alpha=0.8)
    
    ax1.set_xlabel('Precipitation (mm)', fontsize=12)
    ax1.set_ylabel('Mean Biomass at 100 Years', fontsize=12)
    ax1.set_title('Precipitation-Biomass Relationship After 100 Years', fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Add shaded region for training range
    ax1.axvspan(43, 53, alpha=0.2, color='gray', label='Training Range')
    ax1.text(48, ax1.get_ylim()[1]*0.95, 'Training\nRange', ha='center', fontsize=10, style='italic')
    
    # Right plot: Error Analysis
    pde_error = np.abs(pde_biomass - gt_biomass)
    rcnn_error = np.abs(rcnn_biomass - gt_biomass)
    
    ax2.plot(precip_values, pde_error, 's--', color=colors['PDE'], 
             linewidth=2, markersize=7, label='PDE Error', alpha=0.8)
    ax2.plot(precip_values, rcnn_error, '^:', color=colors['RCNN'], 
             linewidth=2, markersize=7, label='RCNN Error', alpha=0.8)
    
    ax2.set_xlabel('Precipitation (mm)', fontsize=12)
    ax2.set_ylabel('Absolute Error in Mean Biomass', fontsize=12)
    ax2.set_title('Model Error vs Precipitation', fontsize=14, fontweight='bold')
    ax2.legend(loc='best', fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    # Add shaded region for training range
    ax2.axvspan(43, 53, alpha=0.2, color='gray')
    
    # Add annotations for extrapolation regions
    ax2.text(35, ax2.get_ylim()[1]*0.9, '← Extrapolation', fontsize=10, style='italic', color='red')
    ax2.text(65, ax2.get_ylim()[1]*0.9, 'Extrapolation →', fontsize=10, style='italic', color='red')
    
    plt.suptitle('Model Performance Across Precipitation Range (100-Year Simulation)', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'precipitation_biomass_relationship.png'), 
                dpi=150, bbox_inches='tight')
    plt.show()
    
    # Print statistics
    print("\n" + "="*60)
    print("PRECIPITATION-BIOMASS RELATIONSHIP ANALYSIS")
    print("="*60)
    
    # Calculate correlations
    pde_corr = np.corrcoef(gt_biomass, pde_biomass)[0, 1]
    rcnn_corr = np.corrcoef(gt_biomass, rcnn_biomass)[0, 1]
    
    print(f"Correlation with Ground Truth:")
    print(f"  PDE Model:  {pde_corr:.4f}")
    print(f"  RCNN Model: {rcnn_corr:.4f}")
    
    # Calculate errors in different regions
    training_mask = (precip_values >= 43) & (precip_values <= 53)
    extrap_low_mask = precip_values < 43
    extrap_high_mask = precip_values > 53
    
    if np.any(training_mask):
        print(f"\nMean Absolute Error in Training Range (43-53 mm):")
        print(f"  PDE Model:  {np.mean(pde_error[training_mask]):.4f}")
        print(f"  RCNN Model: {np.mean(rcnn_error[training_mask]):.4f}")
    
    if np.any(extrap_low_mask):
        print(f"\nMean Absolute Error in Low Extrapolation (<43 mm):")
        print(f"  PDE Model:  {np.mean(pde_error[extrap_low_mask]):.4f}")
        print(f"  RCNN Model: {np.mean(rcnn_error[extrap_low_mask]):.4f}")
    
    if np.any(extrap_high_mask):
        print(f"\nMean Absolute Error in High Extrapolation (>53 mm):")
        print(f"  PDE Model:  {np.mean(pde_error[extrap_high_mask]):.4f}")
        print(f"  RCNN Model: {np.mean(rcnn_error[extrap_high_mask]):.4f}")

def create_detailed_spatial_comparison(models, device, save_dir):
    """
    Create detailed 2D spatial comparison at 100 years for selected precipitation values
    This function actually runs the simulations to get spatial patterns
    """
    
    print("\nGenerating detailed spatial comparison at 100 years...")
    
    # Generate initial equilibrium state
    data_generator = SyntheticDataGenerator(grid_size=(150, 150), device=device)
    equilibrium_state = data_generator.generate_equilibrium_state(
        equilibrium_precipitation=43.0, equilibrium_steps=2500
    )
    
    # Select representative precipitation values
    selected_precips = [30.0, 45.0, 60.0]  # Low, medium (in training range), high
    time_steps = 100  # 100 years
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    fig.suptitle('Spatial Biomass Patterns at 100 Years', fontsize=16, fontweight='bold', y=1.02)
    
    # Color map
    cmap = 'YlGn'
    vmin, vmax = 0, 2.0  # Biomass range
    
    for col_idx, precip_val in enumerate(selected_precips):
        print(f"  Processing precipitation: {precip_val} mm")
        
        # Convert to tensor
        precip_tensor = torch.tensor([precip_val])
        
        # Generate ground truth
        gt_result = generate_ground_truth(equilibrium_state, precip_tensor, time_steps, device)
        gt_final = gt_result[0][-1][0, 0].cpu().numpy()
        
        # Run PDE model
        pde_result = run_pde_simulation(models['pde_model'], equilibrium_state, 
                                       precip_tensor, time_steps, device)
        pde_final = pde_result[0][-1][0, 0].detach().cpu().numpy()
        
        # Run RCNN model
        rcnn_result = run_rcnn_simulation(models['rcnn_model'], equilibrium_state,
                                         precip_tensor, time_steps, device)
        rcnn_final = rcnn_result[0][-1][0, 0].detach().cpu().numpy()
        
        # Plot Ground Truth
        im0 = axes[0, col_idx].imshow(gt_final, cmap=cmap, vmin=vmin, vmax=vmax)
        axes[0, col_idx].set_title(f'Precip: {precip_val} mm\nMean: {np.mean(gt_final):.3f}', 
                                   fontsize=11)
        axes[0, col_idx].axis('off')
        
        # Plot PDE Model
        im1 = axes[1, col_idx].imshow(pde_final, cmap=cmap, vmin=vmin, vmax=vmax)
        pde_error = np.mean((pde_final - gt_final) ** 2)
        axes[1, col_idx].set_title(f'Mean: {np.mean(pde_final):.3f}\nMSE: {pde_error:.4f}', 
                                   fontsize=10)
        axes[1, col_idx].axis('off')
        
        # Plot RCNN Model
        im2 = axes[2, col_idx].imshow(rcnn_final, cmap=cmap, vmin=vmin, vmax=vmax)
        rcnn_error = np.mean((rcnn_final - gt_final) ** 2)
        axes[2, col_idx].set_title(f'Mean: {np.mean(rcnn_final):.3f}\nMSE: {rcnn_error:.4f}', 
                                   fontsize=10)
        axes[2, col_idx].axis('off')
        
        # Add colorbars for the last column
        if col_idx == 2:
            plt.colorbar(im0, ax=axes[0, col_idx], fraction=0.046, pad=0.04)
            plt.colorbar(im1, ax=axes[1, col_idx], fraction=0.046, pad=0.04)
            plt.colorbar(im2, ax=axes[2, col_idx], fraction=0.046, pad=0.04)
    
    # Add row labels
    row_labels = ['Ground Truth', 'PDE Model', 'RCNN Model']
    for ax, label in zip(axes[:, 0], row_labels):
        ax.text(-0.15, 0.5, label, transform=ax.transAxes, fontsize=12,
                fontweight='bold', va='center', ha='right', rotation=90)
    
    # Add precipitation range indicators
    #fig.text(0.28, 0.02, '← Extrapolation', fontsize=10, ha='center', color='red', style='italic')
    #fig.text(0.5, 0.02, 'Training Range', fontsize=10, ha='center', color='green', style='italic')
    #fig.text(0.72, 0.02, 'Extrapolation →', fontsize=10, ha='center', color='red', style='italic')
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.05)
    plt.savefig(os.path.join(save_dir, 'spatial_comparison_100_years_detailed.png'), 
                dpi=150, bbox_inches='tight')
    plt.show()
    
    print("  Spatial comparison complete!")
#%%
def main():
    """Main function to run the complete comparison analysis"""
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Paths to model directories
    pde_dir = r"./results/synthetic/synthetic_results_4site"
    rcnn_dir = r"./results/synthetic/synthetic_rcnn_results_4site"
    save_dir = r"./results\synthetic/model_comparison_results"
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    print("Loading best models from both approaches...")
    models = load_best_models(pde_dir, rcnn_dir, device)
    
    print("\nModel Information:")
    print(f"PDE Model: ID {models['pde_info']['id']}, Training Loss: {models['pde_info']['loss']:.6f}")
    print(f"RCNN Model: ID {models['rcnn_info']['id']}, Training Loss: {models['rcnn_info']['loss']:.6f}")
    print(f"RCNN Config: {models['rcnn_info']['config']}")
    
    print("\nRunning extrapolation experiments...")
    experiments = run_extrapolation_experiments(models, device)
    
    print("\nCreating visualizations...")
    visualize_results(experiments, models, save_dir)
    
    # Add the new visualizations here
    print("\nCreating enhanced visualizations...")
    
    # 1. Precipitation-Biomass Relationship Plot
    create_precipitation_biomass_relationship(experiments, save_dir)
    
    # 2. Detailed Spatial Comparison at 100 years
    create_detailed_spatial_comparison(models, device, save_dir)
    
    # Save experiment results
    print("\nSaving experiment data...")
    
    # Convert tensors to numpy for JSON serialization
    def tensor_to_list(obj):
        if isinstance(obj, torch.Tensor):
            return obj.cpu().numpy().tolist()
        elif isinstance(obj, list):
            return [tensor_to_list(item) for item in obj]
        elif isinstance(obj, dict):
            return {key: tensor_to_list(value) for key, value in obj.items()}
        else:
            return obj
    
    # Save summary results
    summary = {
        'pde_model_info': models['pde_info'],
        'rcnn_model_info': models['rcnn_info'],
        'experiments': {}
    }
    
    for exp_name, exp_data in experiments.items():
        if exp_name == 'precipitation_range':
            # Special handling for precipitation_range
            summary['experiments'][exp_name] = {
                'description': exp_data['description'],
                'precipitation_values': exp_data['precipitation_values'].tolist(),
                'pde_final_biomass': exp_data['pde_final_biomass'],
                'rcnn_final_biomass': exp_data['rcnn_final_biomass'],
                'gt_final_biomass': exp_data['gt_final_biomass']
            }
        else:
            # Compute final metrics for other experiments
            pde_metrics = compute_extrapolation_metrics(exp_data['pde_predictions'], exp_data['ground_truth'])
            rcnn_metrics = compute_extrapolation_metrics(exp_data['rcnn_predictions'], exp_data['ground_truth'])
            
            final_steps = 3
            pde_final_mse = float(np.mean([np.mean(site_mse[-final_steps:]) for site_mse in pde_metrics['mse']]))
            rcnn_final_mse = float(np.mean([np.mean(site_mse[-final_steps:]) for site_mse in rcnn_metrics['mse']]))
            
            pde_final_corr = float(np.mean([np.mean(site_corr[-final_steps:]) for site_corr in pde_metrics['spatial_correlation']]))
            rcnn_final_corr = float(np.mean([np.mean(site_corr[-final_steps:]) for site_corr in rcnn_metrics['spatial_correlation']]))
            
            summary['experiments'][exp_name] = {
                'description': exp_data['description'],
                'precipitation_values': exp_data['precipitation_values'].tolist(),
                'pde_final_mse': pde_final_mse,
                'rcnn_final_mse': rcnn_final_mse,
                'pde_final_correlation': pde_final_corr,
                'rcnn_final_correlation': rcnn_final_corr,
                'pde_better': pde_final_mse < rcnn_final_mse
            }
    
    with open(os.path.join(save_dir, 'extrapolation_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nAnalysis complete! Results saved to: {save_dir}")
    print("\nGenerated files:")
    print("- temporal_extrapolation_analysis.png")
    print("- precipitation_extrapolation_analysis.png") 
    print("- combined_extrapolation_analysis.png")
    print("- precipitation_biomass_relationship.png")
    print("- spatial_comparison_100_years_detailed.png")
    print("- extrapolation_summary.json")
    
    # Final summary
    print("\n" + "="*60)
    print("FINAL EXTRAPOLATION COMPARISON SUMMARY")
    print("="*60)
    
    return summary

if __name__ == "__main__":
    summary = main()
# %%
