# -*- coding: utf-8 -*-
"""
Train models using time series of satellite images from multiple locations
with corresponding annual precipitation measurements
"""
#%%
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import math
import random
import numpy as np
from tqdm import tqdm
import os
import pickle
import json
from typing import Tuple, List, Optional, Dict
from PIL import Image
import rasterio
from rasterio.windows import Window
import pandas as pd
from sklearn.preprocessing import StandardScaler
import glob

# Set up reproducibility
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# Model parameters (kept similar to original)
class EcologicalParameters:
    # Diffusion coefficients
    SURFACE_WATER_DIFFUSION = 8.0
    SOIL_WATER_DIFFUSION = 1.0
    BIOMASS_DIFFUSION = 0.05

    # Loss rates
    EVAPORATION_RATE = 0.3
    SEEPAGE_RATE = 0.4
    MORTALITY_RATE = 0.6

    # Water cycle rates
    INFILTRATION_RATE = 2.1
    PLANT_UPTAKE_RATE = 1.9

    # Biological parameters
    BASE_PRECIPITATION = 0.675
    WATER_USE_EFFICIENCY = 0.55
    GROWTH_FACTOR_ETA = 0
    GROWTH_EXPONENT_Q = 0
    TIME_STEP = 0.005

    # Training parameters
    NOISE_LEVEL = 0.05

# Core NCA Model with Adaptive Time Stepping
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

        # Create diffusion operators
        for name in ['surface_water', 'soil_water', 'biomass']:
            conv = nn.Conv2d(1, 1, kernel_size=3, padding=1,
                           padding_mode="replicate", bias=False)
            conv.weight = nn.Parameter(laplacian_kernel[None, None, :], requires_grad=False)
            setattr(self, f"{name}_diffusion_op", conv)

    def _initialize_parameters(self):
        if self.trainable:
            
            self.precipitation = EcologicalParameters.BASE_PRECIPITATION

            # Initialize with random values
            self.infiltration_rate = nn.Parameter(torch.rand(1))
            self.seepage_rate = nn.Parameter(torch.rand(1))
            self.plant_uptake_rate = nn.Parameter(torch.rand(1))
            self.mortality_rate = nn.Parameter(torch.rand(1))
            self.evaporation_rate = nn.Parameter(torch.rand(1))
            self.water_use_efficiency = nn.Parameter(torch.rand(1))

            self.surface_water_diffusion_coeff = nn.Parameter(torch.rand(1))
            self.soil_water_diffusion_coeff = nn.Parameter(torch.rand(1))
            self.biomass_diffusion_coeff = nn.Parameter(torch.rand(1))

            # Fixed nonlinearity parameters
            self.growth_factor_eta = EcologicalParameters.GROWTH_FACTOR_ETA
            self.growth_exponent_q = EcologicalParameters.GROWTH_EXPONENT_Q
        else:
            # Fixed parameters
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
                biomass: torch.Tensor, precipitation_rate: float = 0.0,
                time_step: float = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass with configurable precipitation and time step
        
        Args:
            surface_water, soil_water, biomass: State tensors
            precipitation_rate: Precipitation rate for this step
            time_step: Time step size (uses base_time_step if None)
        """
        if time_step is None:
            time_step = self.base_time_step
            
        growth_term = (1 + self.growth_factor_eta * biomass ** self.growth_exponent_q)

        surface_water = surface_water + (
            self.surface_water_diffusion_coeff * self.surface_water_diffusion_op(surface_water)
            - self.evaporation_rate * surface_water
            + precipitation_rate  # Use passed precipitation rate instead of self.precipitation
            - self.infiltration_rate * surface_water * biomass
        ) * time_step

        soil_water = soil_water + (
            self.soil_water_diffusion_coeff * self.soil_water_diffusion_op(soil_water)
            - self.seepage_rate * soil_water
            + self.infiltration_rate * surface_water * biomass
            - self.plant_uptake_rate * soil_water * biomass * growth_term
        ) * time_step

        biomass = biomass + (
            self.biomass_diffusion_coeff * self.biomass_diffusion_op(biomass)
            - self.mortality_rate * biomass
            + self.water_use_efficiency * self.plant_uptake_rate * soil_water * biomass * growth_term
        ) * time_step

        return surface_water, soil_water, biomass

    def simulate_year_adaptive(self, surface_water: torch.Tensor, soil_water: torch.Tensor,
                              biomass: torch.Tensor, annual_precipitation: float,
                              rain_day: int = 91, fine_period_days: int = 30,
                              fine_timestep_factor: int = 10) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Simulate one year with adaptive time stepping after rain event
        
        Args:
            surface_water, soil_water, biomass: Initial state tensors
            annual_precipitation: Total precipitation for the year
            rain_day: Day when rain occurs (0-181, where 182 days = 1 year)
            fine_period_days: Number of days after rain to use fine time steps
            fine_timestep_factor: Factor by which to reduce time step (higher = finer)
        
        Returns:
            Final state tensors after one year simulation
        """
        steps_per_year = 102
        current_surface_water = surface_water.clone()
        current_soil_water = soil_water.clone()
        current_biomass = biomass.clone()
        
        for day in range(steps_per_year):
            # Determine if this is the rain day
            if day == rain_day:
                # Apply all annual precipitation in one step
                precipitation_rate = annual_precipitation
            else:
                precipitation_rate = 0.0
            
            # Determine time step size
            if rain_day <= day < rain_day + fine_period_days:
                # Use fine time step during the period after rain
                fine_time_step = self.base_time_step / fine_timestep_factor
                
                # Run multiple sub-steps to maintain temporal accuracy
                for sub_step in range(fine_timestep_factor):
                    # Only apply precipitation on the first sub-step of rain day
                    sub_precip = precipitation_rate/fine_time_step if (day == rain_day and sub_step == 0) else 0.0
                    
                    current_surface_water, current_soil_water, current_biomass = self.forward(
                        current_surface_water, current_soil_water, current_biomass,
                        precipitation_rate=sub_precip,
                        time_step=fine_time_step
                    )
            else:
                # Use normal time step
                current_surface_water, current_soil_water, current_biomass = self.forward(
                    current_surface_water, current_soil_water, current_biomass,
                    precipitation_rate=precipitation_rate,
                    time_step=self.base_time_step
                )
        
        return current_surface_water, current_soil_water, current_biomass

    def simulate_year_weekly(self, surface_water: torch.Tensor, soil_water: torch.Tensor,
                             biomass: torch.Tensor, weekly_precipitation: list,
                             steps_per_week: int = 3) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Simulate one year with weekly precipitation forcing.
        Forces the total simulated time for one year to be exactly 1.0.
        """
        current_surface_water = surface_water.clone()
        current_soil_water = soil_water.clone()
        current_biomass = biomass.clone()

        num_weeks = len(weekly_precipitation)
        total_steps_per_year = num_weeks * steps_per_week
        
        # 1. Enforce that a full year equals exactly 1.0 time unit
        dt = 1.0 / total_steps_per_year

        for week in range(num_weeks):
            # Calculate physical volume for the week (rate in mm/day * 7 days)
            real_weekly_volume = weekly_precipitation[week] * 7.0
            
            # Calculate volume to inject per simulation step
            volume_per_step = real_weekly_volume / steps_per_week
            
            # 2. Scale the rate for the forward pass so that (rate * dt) = volume_per_step
            adjusted_precip_rate = volume_per_step / dt

            for step in range(steps_per_week):
                current_surface_water, current_soil_water, current_biomass = self.forward(
                    current_surface_water, current_soil_water, current_biomass,
                    precipitation_rate=adjusted_precip_rate,
                    time_step=dt  # Pass the computed dt, overriding self.base_time_step
                )

        return current_surface_water, current_soil_water, current_biomass
# Real Data Loader
class RealDataLoader:
    def __init__(self, data_dir: str, selected_sites: List[str] = None,
                 device: Optional[torch.device] = None,
                 use_weekly_precip: bool = True):
        self.data_dir = data_dir
        self.selected_sites = selected_sites
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_weekly_precip = use_weekly_precip
        self.location_data = {}
        self.scaler = StandardScaler()

        # Load data from selected locations
        self._load_selected_locations()

    def _load_selected_locations(self):
        """Load data from selected location directories"""
        # Get all available subsite directories
        all_dirs = [d for d in os.listdir(self.data_dir)
                   if os.path.isdir(os.path.join(self.data_dir, d)) and d.startswith('subsite_')]

        # If no sites selected, use all available sites
        if self.selected_sites is None:
            location_dirs = all_dirs
            print(f"No sites specified, using all available sites")
        else:
            # Filter directories based on selected sites
            location_dirs = []
            for site in self.selected_sites:
                subsite_name = f'subsite_{site}'
                if subsite_name in all_dirs:
                    location_dirs.append(subsite_name)
                else:
                    print(f"Warning: subsite_{site} not found in {self.data_dir}")

        print(f"Found {len(all_dirs)} total sites: {all_dirs}")
        print(f"Loading {len(location_dirs)} selected sites: {location_dirs}")

        for location in location_dirs:
            location_path = os.path.join(self.data_dir, location)
            try:
                self.location_data[location] = self._load_location_data(location_path, location)
            except Exception as e:
                print(f"Error loading {location}: {e}")
                continue

    def _load_location_data(self, location_path: str, location_name: str) -> Dict:
        """Load data for a single location"""
        print(f"Loading data from {location_path}")

        # Load precipitation data
        precip_dir = os.path.join(location_path, f'{location_name}_precip')
        precip_file = os.path.join(precip_dir, f'{location_name}_precip.csv')

        if not os.path.exists(precip_file):
            raise FileNotFoundError(f"Precipitation file not found: {precip_file}")

        precip_df = pd.read_csv(precip_file)
        print(f"  Precipitation data columns: {list(precip_df.columns)}")

        # Handle the datetime and precipitation columns
        # Assuming datetime column contains year information (e.g., 232013 -> 2013)
        if 'datetime' in precip_df.columns and 'total_precipitation_sum' in precip_df.columns:
            # Extract year from datetime (assuming format like 232013 where last 4 digits are year)
            precip_df['year'] = precip_df['datetime'].astype(str).str[-4:].astype(int)
            precip_dict = dict(zip(precip_df['year'], precip_df['total_precipitation_sum']*1000))
        else:
            # Try alternative column names
            date_cols = [col for col in precip_df.columns if 'date' in col.lower() or 'year' in col.lower()]
            precip_cols = [col for col in precip_df.columns if 'precip' in col.lower() or 'rain' in col.lower()]

            if len(date_cols) > 0 and len(precip_cols) > 0:
                date_col = date_cols[0]
                precip_col = precip_cols[0]

                # Extract year if needed
                if precip_df[date_col].dtype == 'object' or precip_df[date_col].max() > 3000:
                    precip_df['year'] = precip_df[date_col].astype(str).str[-4:].astype(int)
                else:
                    precip_df['year'] = precip_df[date_col]

                precip_dict = dict(zip(precip_df['year'], precip_df[precip_col]))
            else:
                raise ValueError(f"Could not identify date and precipitation columns in {precip_file}")

        # Load weekly precipitation if requested
        weekly_precip_dict = {}
        if self.use_weekly_precip:
            weekly_precip_file = os.path.join(precip_dir, f'{location_name}_weekly_precip.csv')
            if os.path.exists(weekly_precip_file):
                weekly_df = pd.read_csv(weekly_precip_file)
                for year in weekly_df['year'].unique():
                    year_data = weekly_df[weekly_df['year'] == year].sort_values('week')
                    weekly_precip_dict[int(year)] = year_data['precipitation_mm_per_day'].tolist()
                print(f"  Loaded weekly precipitation: {len(weekly_precip_dict)} years")
            else:
                print(f"  Warning: Weekly precipitation file not found: {weekly_precip_file}")
                print(f"  Will distribute annual totals evenly across 52 weeks as fallback")

        # Load satellite images
        images_dir = os.path.join(location_path, f'{location_name}_ndvi')
        if not os.path.exists(images_dir):
            raise FileNotFoundError(f"Images directory not found: {images_dir}")

        # Get all image files
        image_files = glob.glob(os.path.join(images_dir, '*.tif'))
        image_files.extend(glob.glob(os.path.join(images_dir, '*.tiff')))

        # Extract years from filenames with multiple naming conventions
        # Extract years from standardized filenames
        image_data = []
        for img_file in image_files:
            filename = os.path.basename(img_file)
            try:
                # Extract 4-digit year from filename (assuming format like "something_YYYY.tif")
                year = int(filename.split('_')[-1].split('.')[0])
                
                if year in precip_dict:
                    entry = {
                        'year': year,
                        'file_path': img_file,
                        'precipitation': precip_dict[year]
                    }
                    if self.use_weekly_precip:
                        if year in weekly_precip_dict:
                            entry['weekly_precipitation'] = weekly_precip_dict[year]
                        else:
                            # Fallback: distribute annual total evenly across 52 weeks
                            annual_mm = precip_dict[year]
                            weekly_rate = annual_mm / 365.0  # mm/day
                            entry['weekly_precipitation'] = [weekly_rate] * 52
                    image_data.append(entry)
                else:
                    print(f"Warning: No precipitation data for year {year} from {filename}")
                    
            except (ValueError, IndexError) as e:
                print(f"Warning: Could not extract year from {filename}: {e}")
                continue

        # Sort by year
        image_data.sort(key=lambda x: x['year'])

        print(f"  Loaded {len(image_data)} images with precipitation data")
        print(f"  Years available: {sorted([img['year'] for img in image_data])}")
        if len(image_data) > 0:
            precip_values = [img['precipitation'] for img in image_data]
            print(f"  Precipitation range: {min(precip_values):.1f} - {max(precip_values):.1f}")
        else:
            print(f"  Warning: No images found with matching precipitation data")

        return {
            'precipitation_data': precip_dict,
            'image_time_series': image_data,
            'location_path': location_path
        }

    def _load_satellite_image(self, file_path: str,
                             ndvi_to_biomass_multiplier: float = 1500.0) -> torch.Tensor:
        """
        Satellite image loading - applies NDVI formula directly to raw band values
        Preserves original image dimensions

        Args:
            file_path: Path to the satellite image
            ndvi_to_biomass_multiplier: Scaling factor to convert NDVI to biomass units
        """
        try:
            with rasterio.open(file_path) as src:
                # Check if we have exactly 2 bands
                if src.count != 2:
                    raise ValueError(f"Expected 2 bands (Red, NIR), found {src.count} bands in {file_path}")

                # Read both bands: assuming band 1 = Red, band 2 = NIR
                red_band = src.read(1).astype(np.float32)
                nir_band = src.read(2).astype(np.float32)

                # Compute NDVI directly: (NIR - Red) / (NIR + Red)
                denominator = nir_band + red_band

                # Handle division by zero
                ndvi = np.zeros_like(denominator)
                valid_mask = denominator > 0  # Only compute where denominator is positive
                ndvi[valid_mask] = (nir_band[valid_mask] - red_band[valid_mask]) / denominator[valid_mask]

                # NDVI ranges from -1 to 1

                # Option 2: Only use positive NDVI values (vegetation)
                biomass = np.maximum(ndvi, 0) * ndvi_to_biomass_multiplier

                # Convert to tensor and add batch and channel dimensions
                tensor = torch.from_numpy(biomass).unsqueeze(0).unsqueeze(0)

                # Ensure non-negative biomass values
                tensor = torch.clamp(tensor, 0, ndvi_to_biomass_multiplier)

                return tensor.to(self.device)

        except Exception as e:
            print(f"Error loading image {file_path}: {e}")
            # Return default values if image loading fails (using arbitrary fallback size)
            return torch.full((1, 1, 100, 100), 0.01, device=self.device)

    def get_training_data(self, ndvi_to_biomass_multiplier: float = 1500.0) -> Dict:
        """
        Prepare training data from all locations
        Preserves original image dimensions

        Args:
            ndvi_to_biomass_multiplier: Scaling factor to convert NDVI to biomass units

        Returns:
            Dict containing:
            - location_time_series: Dict[location_name, List[Dict]]
            - precipitation_values: List of all precipitation values
            - years: List of all years
            - image_dimensions: Dict of original image dimensions per location
        """
        training_data = {
            'location_time_series': {},
            'precipitation_stats': {},
            'global_stats': {'min_precip': float('inf'), 'max_precip': 0},
            'ndvi_to_biomass_multiplier': ndvi_to_biomass_multiplier,
            'image_dimensions': {}  # Track original dimensions
        }

        all_precipitations = []
        all_biomass_values = []

        for location_name, location_data in self.location_data.items():
            print(f"Processing {location_name}...")

            time_series = []
            precipitations = []
            biomass_values = []
            location_dimensions = []

            for img_data in location_data['image_time_series']:
                # Load satellite image and compute NDVI -> biomass 
                biomass_image = self._load_satellite_image(
                    img_data['file_path'],
                    ndvi_to_biomass_multiplier
                )

                # Track image dimensions
                img_height, img_width = biomass_image.shape[2], biomass_image.shape[3]
                location_dimensions.append((img_height, img_width))

                # Use precipitation as-is (no unit conversion)
                precipitation = img_data['precipitation']

                entry = {
                    'year': img_data['year'],
                    'biomass': biomass_image,
                    'precipitation': precipitation,
                    'dimensions': (img_height, img_width)
                }
                if 'weekly_precipitation' in img_data:
                    entry['weekly_precipitation'] = img_data['weekly_precipitation']
                time_series.append(entry)

                precipitations.append(precipitation)
                all_precipitations.append(precipitation)

                # Track biomass statistics
                biomass_mean = biomass_image.mean().item()
                biomass_values.append(biomass_mean)
                all_biomass_values.append(biomass_mean)

            # Only add location if it has data
            if len(time_series) > 0 and len(precipitations) > 0:
                training_data['location_time_series'][location_name] = time_series
                training_data['precipitation_stats'][location_name] = {
                    'mean': np.mean(precipitations),
                    'std': np.std(precipitations),
                    'min': np.min(precipitations),
                    'max': np.max(precipitations)
                }

                # Store dimension info for this location
                unique_dims = list(set(location_dimensions))
                training_data['image_dimensions'][location_name] = {
                    'dimensions': unique_dims,
                    'consistent': len(unique_dims) == 1,
                    'total_images': len(location_dimensions)
                }

                print(f"  - {len(time_series)} time points")
                print(f"  - Image dimensions: {unique_dims}")
                if len(unique_dims) > 1:
                    print(f"    Warning: Mixed dimensions found in {location_name}")
                print(f"  - Precipitation range: {np.min(precipitations):.3f} - {np.max(precipitations):.3f}")
                print(f"  - NDVI-derived biomass range: {np.min(biomass_values):.3f} - {np.max(biomass_values):.3f}")

                # Add to global lists
                all_precipitations.extend(precipitations)
                all_biomass_values.extend(biomass_values)
            else:
                print(f"  - Warning: No valid data found for {location_name}, skipping location")

        # Global statistics - only compute if we have data
        if len(all_precipitations) > 0 and len(all_biomass_values) > 0:
            training_data['global_stats'] = {
                'min_precip': np.min(all_precipitations),
                'max_precip': np.max(all_precipitations),
                'mean_precip': np.mean(all_precipitations),
                'std_precip': np.std(all_precipitations),
                'min_biomass': np.min(all_biomass_values),
                'max_biomass': np.max(all_biomass_values),
                'mean_biomass': np.mean(all_biomass_values),
                'std_biomass': np.std(all_biomass_values)
            }
        else:
            print("Warning: No valid data found across all locations!")
            training_data['global_stats'] = {
                'min_precip': 0, 'max_precip': 0, 'mean_precip': 0, 'std_precip': 0,
                'min_biomass': 0, 'max_biomass': 0, 'mean_biomass': 0, 'std_biomass': 0
            }

        print(f"\nSummary:")
        print(f"  - Total locations with valid data: {len(training_data['location_time_series'])}")
        print(f"  - Total time points: {sum(len(ts) for ts in training_data['location_time_series'].values())}")

        return training_data

    def plot_sample_images(self, training_data: Dict, ndvi_to_biomass_multiplier: float = 1500.0):
        """
        Plot one sample image from each location to visualize the data before training
        """
        num_locations = len(training_data['location_time_series'])
        if num_locations == 0:
            print("No data to plot!")
            return

        # Create subplots - arrange in a reasonable grid
        cols = min(4, num_locations)  # Max 4 columns
        rows = (num_locations + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
        if num_locations == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes.reshape(1, -1)

        # Flatten axes for easier indexing
        axes_flat = axes.flatten() if num_locations > 1 else axes

        location_idx = 0
        for location_name, time_series in training_data['location_time_series'].items():
            if len(time_series) == 0:
                continue

            # Get the first (or middle) image from this location
            sample_idx = len(time_series) // 2  # Middle image for better representation
            sample_data = time_series[sample_idx]

            # Extract biomass data and convert to numpy
            biomass_tensor = sample_data['biomass']
            biomass_image = biomass_tensor.squeeze().cpu().numpy()  # Remove batch and channel dims

            # Plot the image
            ax = axes_flat[location_idx]
            im = ax.imshow(biomass_image, cmap='RdYlGn', vmin=0, vmax=ndvi_to_biomass_multiplier)

            # Add colorbar
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            # Set title with location info
            ax.set_title(f'{location_name}\nYear: {sample_data["year"]}\n'
                        f'Precip: {sample_data["precipitation"]:.1f}\n'
                        f'Size: {biomass_image.shape}', fontsize=10)
            ax.axis('off')

            location_idx += 1

        # Hide unused subplots
        for idx in range(location_idx, len(axes_flat)):
            axes_flat[idx].axis('off')

        plt.tight_layout()
        plt.suptitle('Sample NDVI-derived Biomass Images from Each Location',
                    fontsize=14, y=1.02)
        plt.show()

        # Print summary statistics
        print(f"\nSample Image Statistics:")
        for location_name, time_series in training_data['location_time_series'].items():
            if len(time_series) == 0:
                continue
            sample_data = time_series[len(time_series) // 2]
            biomass_image = sample_data['biomass'].squeeze().cpu().numpy()

            print(f"{location_name}:")
            print(f"  - Biomass range: {biomass_image.min():.3f} - {biomass_image.max():.3f}")
            print(f"  - Biomass mean: {biomass_image.mean():.3f}")
            print(f"  - Non-zero pixels: {np.count_nonzero(biomass_image)} / {biomass_image.size} "
                  f"({100*np.count_nonzero(biomass_image)/biomass_image.size:.1f}%)")
            print(f"  - Precipitation: {sample_data['precipitation']:.1f}")
            print(f"  - Year: {sample_data['year']}")
            print()

# Training Function for Real Data with Weekly Precipitation
def train_model_real_data(training_data: Dict,
                          num_epochs: int = 500,
                          learning_rate: float = 0.001,
                          device: torch.device = None,
                          seed: int = None,
                          save_interval: int = 10,
                          steps_per_week: int = 7,
                          use_delta_loss: bool = True) -> Tuple[List[float], invRietkerk, dict, List[dict]]:
    """
    Train model on real satellite data using weekly precipitation forcing.

    Args:
        training_data: Dictionary containing location time series with weekly precipitation
        steps_per_week: Number of integration sub-steps per week (default 7 = ~daily)
        use_delta_loss: If True, use delta-based loss; if False, use absolute value loss
        Other args same as original
    """
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Set different seed for each model if provided
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    # Initialize model with random parameters
    model = invRietkerk(trainable=True).to(device)
    loss_function = nn.MSELoss()

    # Store initial parameters
    initial_params = {}
    for name, param in model.named_parameters():
        if param.data.shape == torch.Size([1]):
            initial_params[name] = param.item()

    # Initialize parameter tracking
    parameter_history = []

    # Adam optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.95),
        eps=1e-8,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.999)
    loss_history = []

    def compute_loss():
        """Loss function using weekly precipitation forcing"""
        total_loss = 0
        num_comparisons = 0

        for location_name, time_series in training_data['location_time_series'].items():
            if len(time_series) == 0:
                continue

            # Initialize states
            initial_biomass = time_series[0]['biomass'].clone()
            pred_surface_water = torch.zeros_like(initial_biomass, device=device)
            pred_soil_water = torch.zeros_like(initial_biomass, device=device)
            pred_biomass = initial_biomass.clone()

            # Simulate through the time series
            for t_idx, time_point in enumerate(time_series[:-1]):  # Exclude last point
                next_time_point = time_series[t_idx + 1]

                # Get observed biomass at current and next time points
                observed_biomass_current = time_point['biomass']
                observed_biomass_next = next_time_point['biomass']

                # Compute observed delta
                observed_delta = observed_biomass_next - observed_biomass_current

                # Store initial biomass state before simulation
                initial_pred_biomass = pred_biomass.clone()

                # Run simulation for one year using weekly precipitation
                weekly_precip = time_point['weekly_precipitation']
                pred_surface_water, pred_soil_water, pred_biomass = model.simulate_year_weekly(
                    pred_surface_water, pred_soil_water, pred_biomass,
                    weekly_precipitation=weekly_precip,
                    steps_per_week=steps_per_week
                )

                if use_delta_loss:
                    # Compute predicted delta
                    predicted_delta = pred_biomass - initial_pred_biomass
                    # Compare predicted delta with observed delta
                    biomass_loss = loss_function(predicted_delta, observed_delta)
                else:
                    # Original loss: compare absolute values
                    biomass_loss = loss_function(pred_biomass, observed_biomass_next)

                total_loss += biomass_loss
                num_comparisons += 1

        # Average loss across all comparisons
        if num_comparisons > 0:
            total_loss = total_loss / num_comparisons

        return total_loss

    # Function to capture current parameters
    def capture_parameters(epoch):
        params = {'epoch': epoch}
        for name, param in model.named_parameters():
            if param.data.shape == torch.Size([1]):
                params[name] = param.item()
        return params

    print(f"Starting training with weekly precipitation:")
    print(f"- Steps per week: {steps_per_week}")
    print(f"- Total steps per year: {52 * steps_per_week}")
    print(f"- Base time step: {EcologicalParameters.TIME_STEP}")
    print(f"- Using {'delta-based' if use_delta_loss else 'absolute value'} loss function")
    print(f"- Total observations: {sum(len(ts) for ts in training_data['location_time_series'].values())}")

    # Training loop
    for epoch in range(num_epochs):
        optimizer.zero_grad()

        # Forward pass and loss computation
        loss = compute_loss()

        # Check for NaN loss
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"Invalid loss encountered at epoch {epoch}: {loss}")
            break

        # Backward pass
        loss.backward()

        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)

        # Optimizer step
        optimizer.step()
        scheduler.step()

        # Clamp parameters to positive values
        with torch.no_grad():
            for param in model.parameters():
                if param.data.shape == torch.Size([1]):
                    param.data.clamp_(0.0001, 10000)  # Small minimum to avoid zeros

        loss_value = loss.item()
        loss_history.append(loss_value)

        # Save parameters every save_interval epochs
        if epoch % save_interval == 0:
            parameter_history.append(capture_parameters(epoch))

        # Print progress every epoch
        if epoch % 100 == 0 or epoch == num_epochs - 1:
            print(f"Epoch {epoch:4d}, Loss: {loss_value:.6f}", flush=True)

            print("Current model parameters:")
            for name, param in model.named_parameters():
                if param.data.shape == torch.Size([1]):
                    print(f"  {name}: {param.item():.4f}", flush=True)

    # Capture final parameters if not already captured
    if (num_epochs - 1) % save_interval != 0:
        parameter_history.append(capture_parameters(num_epochs - 1))

    return loss_history, model, initial_params, parameter_history


# Updated main training function
def train_models_real_data(data_dir: str,
                           selected_sites: List[str] = None,
                           num_models: int = 30,
                           save_dir: str = "",
                           ndvi_to_biomass_multiplier: float = 1500.0,
                           use_delta_loss: bool = True,
                           steps_per_week: int = 7,
                           use_weekly_precip: bool = True):
    """
    Train multiple models with weekly precipitation forcing using real satellite data.

    Args:
        steps_per_week: Number of integration sub-steps per week (default 7 = ~daily)
        use_weekly_precip: If True, load and use weekly precipitation CSVs
        Other args same as original
    """

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create directories
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(f"{save_dir}/models", exist_ok=True)
    os.makedirs(f"{save_dir}/results", exist_ok=True)
    os.makedirs(f"{save_dir}/parameters", exist_ok=True)

    # Load real data with site selection
    print("Loading real satellite data...")
    data_loader = RealDataLoader(data_dir, selected_sites=selected_sites,
                                 device=device, use_weekly_precip=use_weekly_precip)
    training_data = data_loader.get_training_data(
        ndvi_to_biomass_multiplier=ndvi_to_biomass_multiplier
    )

    # Check if we have any valid training data
    if len(training_data['location_time_series']) == 0:
        raise ValueError("No valid training data found!")

    print(f"\nWeekly Precipitation Configuration:")
    print(f"- Steps per week: {steps_per_week}")
    print(f"- Total steps per year: {52 * steps_per_week}")
    print(f"- Base time step: {EcologicalParameters.TIME_STEP}")

    print(f"\nData Summary:")
    print(f"- Selected sites: {selected_sites if selected_sites else 'All available'}")
    print(f"- Locations loaded: {len(training_data['location_time_series'])}")
    print(f"- NDVI to biomass multiplier: {ndvi_to_biomass_multiplier}")
    print(f"- Loss function: {'Delta-based' if use_delta_loss else 'Absolute value'}")

    # Save training data info
    with open(f"{save_dir}/data_info.json", 'w') as f:
        data_info = {
            'selected_sites': selected_sites,
            'locations': list(training_data['location_time_series'].keys()),
            'precipitation_stats': training_data['precipitation_stats'],
            'global_stats': training_data['global_stats'],
            'ndvi_to_biomass_multiplier': ndvi_to_biomass_multiplier,
            'use_delta_loss': use_delta_loss,
            'weekly_precip_config': {
                'steps_per_week': steps_per_week,
                'total_steps_per_year': 52 * steps_per_week,
                'base_time_step': EcologicalParameters.TIME_STEP,
                'use_weekly_precip': use_weekly_precip,
            },
            'time_points_per_location': {
                loc: len(ts) for loc, ts in training_data['location_time_series'].items()
            }
        }
        json.dump(data_info, f, indent=2)

    # Train multiple models
    results = []
    all_final_losses = []

    print(f"\nTraining {num_models} models with weekly precipitation...")

    for model_idx in range(10, 20):
        print(f"\nTraining model {model_idx + 1}/{num_models}...")

        # Use different seed for each model
        model_seed = 77 + model_idx * 102

        try:
            loss_history, trained_model, initial_params, parameter_history = train_model_real_data(
                training_data=training_data,
                num_epochs=7500,
                learning_rate=0.2,
                device=device,
                seed=model_seed,
                save_interval=10,
                steps_per_week=steps_per_week,
                use_delta_loss=use_delta_loss
            )

            final_loss = loss_history[-1]
            all_final_losses.append(final_loss)

            # Get final parameters
            final_params = {}
            for name, param in trained_model.named_parameters():
                if param.data.shape == torch.Size([1]):
                    final_params[name] = param.item()

            # Store results
            model_result = {
                'model_id': model_idx,
                'seed': model_seed,
                'final_loss': final_loss,
                'epochs': len(loss_history),
                'initial_params': initial_params,
                'final_params': final_params,
                'loss_history': loss_history,
                'parameter_snapshots': len(parameter_history),
                'use_delta_loss': use_delta_loss,
                'weekly_config': {
                    'steps_per_week': steps_per_week,
                    'total_steps_per_year': 52 * steps_per_week,
                }
            }
            results.append(model_result)

            # Save model
            torch.save(trained_model.state_dict(),
                      f"{save_dir}/models/model_{model_idx:02d}.pth")

            # Save parameter history
            with open(f"{save_dir}/parameters/model_{model_idx:02d}_params.pkl", 'wb') as f:
                pickle.dump(parameter_history, f)

            print(f"Model {model_idx + 1}: Final loss = {final_loss:.6f}")

        except Exception as e:
            print(f"Model {model_idx + 1} failed: {e}")
            all_final_losses.append(float('nan'))

    # Save results and create visualizations
    successful_models = [r for r in results if not math.isnan(r['final_loss'])]

    print(f"\n{'='*50}")
    print(f"WEEKLY PRECIPITATION TRAINING COMPLETE")
    print(f"{'='*50}")
    print(f"Successful models: {len(successful_models)}/{num_models}")

    if successful_models:
        final_losses = [r['final_loss'] for r in successful_models]
        print(f"Final loss statistics:")
        print(f"- Mean: {np.mean(final_losses):.6f}")
        print(f"- Std:  {np.std(final_losses):.6f}")
        print(f"- Min:  {np.min(final_losses):.6f}")
        print(f"- Max:  {np.max(final_losses):.6f}")

        # Save detailed results
        with open(f"{save_dir}/results/training_summary.json", 'w') as f:
            summary = {
                'total_models': num_models,
                'successful_models': len(successful_models),
                'selected_sites': selected_sites,
                'optimizer': 'Adam',
                'data_type': 'real_satellite_data_weekly_precip',
                'ndvi_to_biomass_multiplier': ndvi_to_biomass_multiplier,
                'use_delta_loss': use_delta_loss,
                'loss_type': 'delta-based MSE' if use_delta_loss else 'absolute value MSE',
                'weekly_precip_config': {
                    'steps_per_week': steps_per_week,
                    'total_steps_per_year': 52 * steps_per_week,
                    'base_time_step': EcologicalParameters.TIME_STEP,
                },
                'statistics': {
                    'mean_loss': float(np.mean(final_losses)),
                    'std_loss': float(np.std(final_losses)),
                    'min_loss': float(np.min(final_losses)),
                    'max_loss': float(np.max(final_losses))
                }
            }
            json.dump(summary, f, indent=2)

        # Create training visualizations
        plt.figure(figsize=(15, 10))

        # Loss distribution
        plt.subplot(2, 2, 1)
        plt.hist(final_losses, bins=10, alpha=0.7, color='skyblue', edgecolor='black')
        plt.xlabel('Final Loss')
        plt.ylabel('Number of Models')
        plt.title(f'Distribution of Final Losses (Weekly Precip)\n{"Delta-based" if use_delta_loss else "Absolute"} Loss')
        plt.grid(True, alpha=0.3)

        # Training curves
        plt.subplot(2, 2, 2)
        for model in successful_models[:10]:  # Show first 10
            plt.plot(model['loss_history'], alpha=0.7, linewidth=1)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Curves')
        plt.yscale('log')
        plt.grid(True, alpha=0.3)

        # Final loss per model
        plt.subplot(2, 2, 3)
        model_ids = [r['model_id'] for r in successful_models]
        plt.bar(model_ids, final_losses, alpha=0.7, color='skyblue', edgecolor='black')
        plt.xlabel('Model ID')
        plt.ylabel('Final Loss')
        plt.title('Final Loss per Model')
        plt.grid(True, alpha=0.3)

        # Summary text
        plt.subplot(2, 2, 4)
        summary_text = f'''Weekly Precipitation Configuration:

Steps per week: {steps_per_week}
Total steps per year: {52 * steps_per_week}
Base time step: {EcologicalParameters.TIME_STEP:.4f}

Precipitation applied as weekly
averages from ERA5 daily data
(52 weeks per year).'''

        plt.text(0.1, 0.5, summary_text, fontsize=11, verticalalignment='center',
                transform=plt.gca().transAxes, family='monospace')
        plt.axis('off')

        plt.tight_layout()
        plt.savefig(f"{save_dir}/results/weekly_training_analysis.png", dpi=150, bbox_inches='tight')
        plt.show()

    return results, training_data


# Example usage with weekly precipitation
if __name__ == "__main__":
    # Set your data and save directory paths here
    data_directory = r"./data"
    save_directory = r"./results/real_data/models"

    # Example usage with site selection and weekly precipitation
    selected_sites = ['b', 'i', 'c', 'e']

    print("Starting invRietkerk training with weekly precipitation...")
    print(f"Data directory: {data_directory}")
    print(f"Save directory: {save_directory}")
    print(f"Selected sites: {selected_sites}")

    # Train models with weekly precipitation from ERA5
    results, training_data = train_models_real_data(
        data_dir=data_directory,
        save_dir=save_directory,
        selected_sites=selected_sites,
        num_models=30,
        ndvi_to_biomass_multiplier=1500,
        use_delta_loss=True,
        steps_per_week=3,  # 7 sub-steps per week (~daily resolution)
        use_weekly_precip=True,  # Use weekly precipitation CSVs
    )

    print("\nWeekly precipitation training completed!")
    print(f"Results saved in: {save_directory}")

# %%
