
#%%
import os
import pickle
import warnings
from typing import Dict, List, Optional, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio

warnings.filterwarnings('ignore')

# Training subsites used for the real-data models
TRAINING_SUBSITES = ['b', 'i', 'c', 'e']
NDVI_TO_BIOMASS_MULTIPLIER = 1500.0

# Ground truth parameters
GROUND_TRUTH_PARAMS = {
    'surface_water_diffusion_coeff': 8.0,
    'soil_water_diffusion_coeff': 1.0, 
    'biomass_diffusion_coeff': 0.05,
    'evaporation_rate': 0.3,
    'seepage_rate': 0.4,
    'mortality_rate': 0.6,
    'infiltration_rate': 2.1,
    'plant_uptake_rate': 1.9,
    'water_use_efficiency': 0.55
}

def compute_mean_training_biomass(data_dir: str,
                                  subsites: List[str] = TRAINING_SUBSITES,
                                  multiplier: float = NDVI_TO_BIOMASS_MULTIPLIER) -> float:
    """Mean biomass over all NDVI images in the given training subsites.

    Biomass is computed exactly as in realdata_train_invPDE._load_satellite_image:
    NDVI = (NIR - Red) / (NIR + Red), then biomass = max(NDVI, 0) * multiplier,
    clamped to [0, multiplier].
    """
    per_image_means = []
    for site in subsites:
        ndvi_dir = os.path.join(data_dir, f'subsite_{site}', f'subsite_{site}_ndvi')
        if not os.path.isdir(ndvi_dir):
            print(f"Warning: NDVI directory not found for subsite {site}: {ndvi_dir}")
            continue
        tif_files = sorted(f for f in os.listdir(ndvi_dir) if f.lower().endswith('.tif'))
        for fname in tif_files:
            fpath = os.path.join(ndvi_dir, fname)
            with rasterio.open(fpath) as src:
                red = src.read(1).astype(np.float32)
                nir = src.read(2).astype(np.float32)
                denom = nir + red
                ndvi = np.zeros_like(denom)
                valid = denom > 0
                ndvi[valid] = (nir[valid] - red[valid]) / denom[valid]
                biomass = np.clip(np.maximum(ndvi, 0) * multiplier, 0, multiplier)
                per_image_means.append(float(biomass.mean()))
    if len(per_image_means) == 0:
        raise RuntimeError(f"No NDVI images found under {data_dir} for subsites {subsites}")
    mean_B = float(np.mean(per_image_means))
    print(f"Mean training biomass over {len(per_image_means)} images "
          f"(subsites {subsites}, multiplier={multiplier}): B = {mean_B:.6f}")
    return mean_B

def compute_composite(param_dic, B):
    """Composite parameter: (r2 B / (l1 + r2 B)) * (r1 B / (l2 + r1 B)) * j,
    where j = water_use_efficiency, r1 = plant_uptake_rate, r2 = infiltration_rate,
    l1 = evaporation_rate, l2 = seepage_rate.
    """
    l1 = param_dic['evaporation_rate']
    l2 = param_dic['seepage_rate']
    r1 = param_dic['plant_uptake_rate']
    r2 = param_dic['infiltration_rate']
    j = param_dic['water_use_efficiency']
    return (r2 * B / (l1 + r2 * B)) * (r1 * B / (l2 + r1 * B)) * j

def compute_turing_evolution(param_dic):
    d1 = param_dic['surface_water_diffusion_coeff']
    d2 = param_dic['soil_water_diffusion_coeff']
    d3 = param_dic['biomass_diffusion_coeff']
    l1 = param_dic['evaporation_rate']
    l2 = param_dic['seepage_rate']
    l3 = param_dic['mortality_rate']
    r1 = param_dic['plant_uptake_rate']
    r2 = param_dic['infiltration_rate']
    
    turing_value = -d1*l2*l3 + d3*(l1+np.sqrt(l1*l2*r2/r1)) * (l2+np.sqrt(l1*l2*r1/r2))
    return turing_value

def load_parameter_history(results_dir: str, model_id: int) -> Optional[List[Dict[str, Any]]]:
    """Load parameter history for a specific model."""
    
    param_dir = os.path.join(results_dir, "parameters")
    param_file = os.path.join(param_dir, f"model_{model_id:02d}_params.pkl")
    
    if not os.path.exists(param_file):
        print(f"Parameter file not found: {param_file}")
        return None
        
    try:
        with open(param_file, 'rb') as f:
            parameter_history = pickle.load(f)
        return parameter_history
    except Exception as e:
        print(f"Failed to load parameter history for model {model_id}: {e}")
        return None

def calculate_parameter_agreement(results_dir: str, max_models: int = 10,
                                  biomass_B: Optional[float] = None) -> pd.DataFrame:
    """Calculate agreement metrics between training runs for each parameter.

    If biomass_B is provided, the composite parameter
    (r2 B / (l1 + r2 B)) * (r1 B / (l2 + r1 B)) * j is also included.
    """

    parameter_names = list(GROUND_TRUTH_PARAMS.keys())

    # Collect final parameter values from all models
    final_params = {param_name: [] for param_name in parameter_names}
    final_params['turing_value'] = []
    if biomass_B is not None:
        final_params['composite_value'] = []
    model_ids = []
    
    models_loaded = 0
    
    for model_id in range(100):  # Try models 0-29 (100 total)
        if models_loaded >= max_models:
            break
            
        param_history = load_parameter_history(results_dir, model_id)
        
        if param_history is None or len(param_history) == 0:
            continue
            
        # Get the final parameter values (last snapshot)
        final_snapshot = param_history[-1]
        
        # Check if all required parameters are present
        if not all(param in final_snapshot for param in parameter_names):
            continue
        
        # Check for invalid values
        if any(value < 0.0011 for param, value in final_snapshot.items() 
               if param in parameter_names):
            continue
        
        # Store final parameter values
        for param_name in parameter_names:
            final_params[param_name].append(final_snapshot[param_name])
        
        # Calculate and store Turing value
        final_params['turing_value'].append(compute_turing_evolution(final_snapshot))
        if biomass_B is not None:
            final_params['composite_value'].append(
                compute_composite(final_snapshot, biomass_B)
            )
        model_ids.append(model_id)
        models_loaded += 1
    
    if models_loaded < 2:
        print("Not enough valid models found for agreement analysis")
        return None
    
    # Create DataFrame with final parameter values
    df_params = pd.DataFrame(final_params, index=model_ids)
    
    # Calculate agreement metrics
    agreement_metrics = {}
    
    extra_cols = ['turing_value']
    if biomass_B is not None:
        extra_cols.append('composite_value')
    for param_name in list(parameter_names) + extra_cols:
        values = df_params[param_name].values
        
        # Basic statistics
        mean_val = np.mean(values)
        std_val = np.std(values)
        cv = std_val / abs(mean_val) if mean_val != 0 else np.inf  # Coefficient of variation
        
        # Range metrics
        min_val = np.min(values)
        max_val = np.max(values)
        range_val = max_val - min_val
        
        # Relative range (range as percentage of mean)
        rel_range = (range_val / abs(mean_val) * 100) if mean_val != 0 else np.inf
        
        # Agreement with ground truth (if available)
        if param_name in GROUND_TRUTH_PARAMS:
            gt_value = GROUND_TRUTH_PARAMS[param_name]
            # Mean absolute percentage error
            mape = np.mean(np.abs((values - gt_value) / gt_value)) * 100
            # Bias (mean relative error)
            bias = np.mean((values - gt_value) / gt_value) * 100
        else:
            mape = np.nan
            bias = np.nan
            gt_value = np.nan
        
        # Pairwise agreement (average correlation with all other runs)
        if len(values) > 1:
            # For final values, we'll use inverse of coefficient of variation as agreement measure
            agreement_score = 1 / (1 + cv) if cv != np.inf else 0
        else:
            agreement_score = 1.0
        
        agreement_metrics[param_name] = {
            'n_models': len(values),
            'mean': mean_val,
            'std': std_val,
            'cv': cv,
            'min': min_val,
            'max': max_val,
            'range': range_val,
            'rel_range_pct': rel_range,
            'ground_truth': gt_value,
            'mape_vs_gt_pct': mape,
            'bias_vs_gt_pct': bias,
            'agreement_score': agreement_score
        }
    
    # Convert to DataFrame for nice display
    agreement_df = pd.DataFrame(agreement_metrics).T
    
    return agreement_df, df_params

def plot_parameter_agreement_summary(agreement_df: pd.DataFrame, save_path: Optional[str] = None):
    """Create visualization of parameter agreement metrics."""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Rows for the main ecological parameters only (exclude derived composites)
    derived_rows = [r for r in ('turing_value', 'composite_value') if r in agreement_df.index]

    # 1. Coefficient of Variation
    ax1 = axes[0, 0]
    cv_values = agreement_df['cv'].drop(derived_rows)  # Exclude derived rows
    param_names = [name.replace('_', ' ').title() for name in cv_values.index]
    
    bars1 = ax1.bar(param_names, cv_values.values, color='skyblue', alpha=0.7)
    ax1.set_title('Parameter Variability Across Runs\n(Coefficient of Variation)', fontweight='bold')
    ax1.set_ylabel('Coefficient of Variation')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, val in zip(bars1, cv_values.values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.3f}', ha='center', va='bottom')
    
    # 2. Agreement Score
    ax2 = axes[0, 1]
    agreement_scores = agreement_df['agreement_score'].drop(derived_rows)
    
    bars2 = ax2.bar(param_names, agreement_scores.values, color='lightcoral', alpha=0.7)
    ax2.set_title('Parameter Agreement Score\n(Higher = More Consistent)', fontweight='bold')
    ax2.set_ylabel('Agreement Score')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)
    
    # Add value labels on bars
    for bar, val in zip(bars2, agreement_scores.values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.3f}', ha='center', va='bottom')
    
    # 3. MAPE vs Ground Truth
    ax3 = axes[1, 0]
    mape_values = agreement_df['mape_vs_gt_pct'].drop(derived_rows).dropna()
    mape_param_names = [name.replace('_', ' ').title() for name in mape_values.index]
    
    bars3 = ax3.bar(mape_param_names, mape_values.values, color='lightgreen', alpha=0.7)
    ax3.set_title('Accuracy vs Ground Truth\n(Mean Absolute Percentage Error)', fontweight='bold')
    ax3.set_ylabel('MAPE (%)')
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, val in zip(bars3, mape_values.values):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}%', ha='center', va='bottom')
    
    # 4. Relative Range
    ax4 = axes[1, 1]
    rel_range_values = agreement_df['rel_range_pct'].drop(derived_rows)
    
    bars4 = ax4.bar(param_names, rel_range_values.values, color='orange', alpha=0.7)
    ax4.set_title('Parameter Range Across Runs\n(Relative to Mean)', fontweight='bold')
    ax4.set_ylabel('Relative Range (%)')
    ax4.tick_params(axis='x', rotation=45)
    ax4.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, val in zip(bars4, rel_range_values.values):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(f"{save_path}_agreement_summary.png", dpi=300, bbox_inches='tight')
        print(f"Agreement summary plot saved to: {save_path}_agreement_summary.png")
    
    plt.show()

def display_agreement_table(agreement_df: pd.DataFrame, save_path: Optional[str] = None):
    """Display and save the parameter agreement table."""
    
    # Create a more readable version of the table
    display_df = agreement_df.copy()
    
    # Round numerical values for better display
    numeric_columns = ['mean', 'std', 'cv', 'min', 'max', 'range', 'rel_range_pct', 
                      'mape_vs_gt_pct', 'bias_vs_gt_pct', 'agreement_score', 'ground_truth']
    
    for col in numeric_columns:
        if col in display_df.columns:
            display_df[col] = display_df[col].round(4)
    
    # Rename columns for better readability
    column_mapping = {
        'n_models': 'N Models',
        'mean': 'Mean',
        'std': 'Std Dev',
        'cv': 'CV',
        'min': 'Min',
        'max': 'Max',
        'range': 'Range',
        'rel_range_pct': 'Rel Range (%)',
        'ground_truth': 'Ground Truth',
        'mape_vs_gt_pct': 'MAPE vs GT (%)',
        'bias_vs_gt_pct': 'Bias vs GT (%)',
        'agreement_score': 'Agreement Score'
    }
    
    display_df = display_df.rename(columns=column_mapping)
    
    # Create a nice parameter name index
    display_df.index = [name.replace('_', ' ').title() for name in display_df.index]
    
    print("\n" + "="*100)
    print("PARAMETER AGREEMENT ANALYSIS")
    print("="*100)
    print(f"Analysis based on final parameter values from {display_df.iloc[0]['N Models']} training runs")
    print("\nMetrics explanation:")
    print("- CV: Coefficient of Variation (lower = more consistent)")
    print("- Rel Range (%): Parameter range as percentage of mean")
    print("- MAPE vs GT (%): Mean Absolute Percentage Error vs Ground Truth")
    print("- Agreement Score: 0-1 scale (higher = more consistent)")
    print("-"*100)
    
    # Display the table
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', 20)
    
    print(display_df.to_string())
    
    # Save to CSV if requested
    if save_path:
        csv_path = f"{save_path}_agreement_table.csv"
        display_df.to_csv(csv_path)
        print(f"\nAgreement table saved to: {csv_path}")
    
    # Reset pandas display options
    pd.reset_option('display.max_columns')
    pd.reset_option('display.width')
    pd.reset_option('display.max_colwidth')
    
    return display_df

def plot_parameter_histories(results_dir: str, max_models: int = 10, save_path: Optional[str] = None) -> int:
    """Plot parameter evolution during training for multiple models."""
    
    parameter_names = list(GROUND_TRUTH_PARAMS.keys())
    
    # Create subplots
    fig, axes = plt.subplots(3, 3, figsize=(20, 16))
    axes = axes.flatten()
    
    # Colors for different models
    colors = plt.cm.tab10(np.linspace(0, 1, max_models))
    
    models_loaded = 0
    
    for model_id in range(100):  # Try models 0-29 (100 total)
        if models_loaded >= max_models:
            break
            
        param_history = load_parameter_history(results_dir, model_id)
        
        if param_history is None or len(param_history) == 0:
            continue
            
        # Extract epochs and parameter values
        epochs = []
        param_values = {param_name: [] for param_name in parameter_names}
        
        for snapshot in param_history:
            if 'epoch' in snapshot:
                epochs.append(snapshot['epoch'])
                
                for param_name in parameter_names:
                    if param_name in snapshot:
                        param_values[param_name].append(snapshot[param_name])
                        
                    else:
                        param_values[param_name].append(np.nan)
            
        if len(epochs) == 0 or any(value < 0.0011 for value in snapshot.values()):
            continue
            
        # Plot each parameter
        for i, param_name in enumerate(parameter_names):
            ax = axes[i]
            
            if len(param_values[param_name]) > 0:
                # Plot parameter evolution
                ax.plot(epochs, param_values[param_name], 
                       color=colors[models_loaded], alpha=0.7, linewidth=1.5,
                       label=f'Model {model_id}' if i == 0 else "")
                
                
        
        models_loaded += 1
    axfontscaler = 1.5
    # Customize each subplot
    for i, param_name in enumerate(parameter_names):
        ax = axes[i]
        ax.set_title(f'{param_name.replace("_", " ").title()}', fontsize=13*axfontscaler, fontweight='bold')
        ax.set_xlabel('Training Epoch', fontsize=12*axfontscaler)
        ax.set_ylabel('Parameter Value', fontsize=12*axfontscaler)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='both', labelsize=10*axfontscaler)
        
        
    
    plt.suptitle(f'Parameter evolution during training', 
                 fontsize=16*axfontscaler, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(f"{save_path}_parameter_histories.png", dpi=300, bbox_inches='tight')
        print(f"Parameter history plot saved to: {save_path}_parameter_histories.png")
    
    plt.show()
    

    #### plot turing evolution

    models_loaded = 0
    
    for model_id in range(100):  # Try models 0-29 (100 total)
        if models_loaded >= max_models:
            break
            
        param_history = load_parameter_history(results_dir, model_id)
        
        if param_history is None or len(param_history) == 0:
            continue
            
        # Extract epochs and parameter values
        epochs = []
        param_values = {param_name: [] for param_name in parameter_names}
        param_values['turing_value'] = []
        for snapshot in param_history:
            if 'epoch' in snapshot:
                epochs.append(snapshot['epoch'])
                
                for param_name in parameter_names:
                    if param_name in snapshot:
                        param_values[param_name].append(snapshot[param_name])
                        
                    else:
                        param_values[param_name].append(np.nan)
            param_values['turing_value'].append(compute_turing_evolution(snapshot))
        if len(epochs) == 0:
            continue
            
        
        plt.plot(epochs[::6], param_values['turing_value'][::6], 
                color=colors[models_loaded], alpha=0.7, linewidth=1.5,
                label=f'Model {model_id}' if i == 0 else "")
        plt.title(f'Turing Value Evolution During Training', fontsize=16*axfontscaler, fontweight='bold')
        
        
        models_loaded += 1


    
    plt.ylim(-100, 100)            
    plt.show()            
    
    return models_loaded

def main() -> None:
    """Main function to create parameter history visualizations and agreement analysis."""
    
    # Set path to your results directory
    results_dir = r"./results/real_data"
    data_dir = r"./data"

    # Check if directory exists
    if not os.path.exists(results_dir):
        print(f"Four-site results directory not found: {results_dir}")
        print("Please update the results_dir path in the script.")
        return

    print(f"Analyzing parameter histories from: {results_dir}")

    # Compute mean biomass over all training images for the composite parameter
    try:
        biomass_B = compute_mean_training_biomass(data_dir)
    except Exception as e:
        print(f"Could not compute mean training biomass ({e}); "
              f"composite parameter will be skipped.")
        biomass_B = None
    
    # Create output directory
    output_dir = os.path.join(os.path.dirname(results_dir), "parameter_history_analysis")
    os.makedirs(output_dir, exist_ok=True)
    
    plot_save_path = os.path.join(output_dir, "four_site")
    
    # Plot parameter evolution during training
    print("\nGenerating parameter history plots...")
    models_shown = plot_parameter_histories(results_dir, max_models=100, save_path=plot_save_path)
    
    if models_shown > 0:
        print(f"Successfully plotted parameter histories for {models_shown} models")
        
        # Calculate and display parameter agreement
        print("\nCalculating parameter agreement metrics...")
        agreement_results = calculate_parameter_agreement(
            results_dir, max_models=100, biomass_B=biomass_B
        )
        
        if agreement_results is not None:
            agreement_df, params_df = agreement_results
            
            # Display the agreement table
            display_agreement_table(agreement_df, save_path=plot_save_path)
            
            # Create agreement summary plots
            print("\nGenerating agreement summary plots...")
            plot_parameter_agreement_summary(agreement_df, save_path=plot_save_path)
            
            # Save the raw parameter values for further analysis
            params_csv_path = f"{plot_save_path}_final_parameter_values.csv"
            params_df.to_csv(params_csv_path)
            print(f"Final parameter values saved to: {params_csv_path}")
        
        print(f"All plots saved to: {output_dir}")
    else:
        print("No model parameter histories found to plot")

if __name__ == "__main__":
    main()
# %%
