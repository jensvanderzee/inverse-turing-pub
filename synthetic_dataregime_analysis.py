
""""
Parameter comparison between different data regimes (1-site vs 4-site)
"""
#%%
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import pickle
import warnings
warnings.filterwarnings('ignore')

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

def load_regime_parameters(results_dir: str, regime_name: str):
    """Load final parameters from all models in a regime."""
    
    param_dir = os.path.join(results_dir, "parameters")
    if not os.path.exists(param_dir):
        print(f"Parameters directory not found for {regime_name}: {param_dir}")
        return {}
        
    regime_params = {}
    loaded_models = 0
    
    for model_id in range(30):
        param_file = os.path.join(param_dir, f"model_{model_id:02d}_params.pkl")
        
        if os.path.exists(param_file):
            try:
                with open(param_file, 'rb') as f:
                    parameter_history = pickle.load(f)
                
                if parameter_history:
                    # Get final parameters (last snapshot)
                    final_params = parameter_history[-1].copy()
                    final_params.pop('epoch', None)
                    
                    # Store parameters by name
                    for param_name, param_value in final_params.items():
                        if param_name not in regime_params:
                            regime_params[param_name] = []
                        regime_params[param_name].append(param_value)
                    
                    loaded_models += 1
                    
            except Exception as e:
                print(f"Failed to load {regime_name} model {model_id}: {e}")
                
    print(f"Loaded {loaded_models} models from {regime_name} regime")
    return regime_params

def calculate_relative_error(estimated_values, true_value):
    """Calculate relative error for a set of estimated values."""
    estimated_values = np.array(estimated_values)
    return np.abs(estimated_values - true_value) / np.abs(true_value) * 100

def create_regime_boxplots(regime1_params, regime2_params, regime1_name="One site", regime2_name="Four sites", save_path=None):
    """Create side-by-side boxplots comparing parameter estimates between regimes."""
    
    parameter_names = list(GROUND_TRUTH_PARAMS.keys())
    
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    axes = axes.flatten()
    
    for i, param_name in enumerate(parameter_names):
        ax = axes[i]
        
        # Get parameter values for both regimes
        regime1_vals = regime1_params.get(param_name, [])
        regime2_vals = regime2_params.get(param_name, [])
        
        if not regime1_vals or not regime2_vals:
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{param_name.replace("_", " ").title()}')
            continue
        
        # Calculate relative errors
        true_val = GROUND_TRUTH_PARAMS[param_name]
        regime1_rel_errors = calculate_relative_error(regime1_vals, true_val)
        regime2_rel_errors = calculate_relative_error(regime2_vals, true_val)
        
        # Create boxplots for relative errors
        box_data = [regime1_rel_errors, regime2_rel_errors]
        labels = [regime1_name, regime2_name]
        
        bp = ax.boxplot(box_data, labels=labels, patch_artist=True, widths=0.6, showfliers=False)
        
        # Customize appearance
        colors = ['lightblue', 'lightgreen']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor('none')
            patch.set_edgecolor('black')
        
        # Make median lines more visible
        for median in bp['medians']:
            median.set_color('red')
            median.set_linewidth(2)
        
        # Add individual points (jittered)
        for j, (vals, label) in enumerate(zip(box_data, labels)):
            x_jitter = np.random.normal(j+1, 0.04, size=len(vals))
            ax.scatter(x_jitter, vals, alpha=0.6, s=15, color='darkblue' if j == 0 else 'darkgreen', zorder=3)
        
        ax.set_ylabel('Relative Error (%)')
        ax.set_title(f'{param_name.replace("_", " ").title()}', fontsize=18)
        ax.grid(True, alpha=0.3, axis='y')
        ax.tick_params(axis='x', labelsize=16)
        ax.tick_params(axis='y', labelsize=16)
        
        # Add zero line (perfect estimation)
        ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        
        # Add text showing mean relative errors
        mean1 = np.mean(regime1_rel_errors)
        mean2 = np.mean(regime2_rel_errors)
        stats_text = f'{regime1_name}: {mean1:.1f}%\n{regime2_name}: {mean2:.1f}%'
        ax.text(0.98, 0.98, stats_text, transform=ax.transAxes, 
               verticalalignment='top', horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
               fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(f"{save_path}_regime_comparison.png", dpi=300, bbox_inches='tight')
    
    plt.show()

def print_simple_summary(regime1_params, regime2_params, regime1_name="1-site", regime2_name="4-site"):
    """Print a simple summary of the comparison using relative error."""
    
    print("=" * 80)
    print(f"PARAMETER ESTIMATION COMPARISON: {regime1_name} vs {regime2_name}")
    print("(Using Relative Error)")
    print("=" * 80)
    
    parameter_names = list(GROUND_TRUTH_PARAMS.keys())
    
    print(f"\nParameter-wise Relative Error Comparison:")
    print("-" * 80)
    
    header = f"{'Parameter':<25} {'True Value':<12} {f'{regime1_name} Rel Err':<15} {f'{regime2_name} Rel Err':<15} {'Improvement':<12}"
    print(header)
    print("-" * len(header))
    
    improvements = []
    
    for param_name in parameter_names:
        regime1_vals = regime1_params.get(param_name, [])
        regime2_vals = regime2_params.get(param_name, [])
        
        if not regime1_vals or not regime2_vals:
            continue
            
        true_val = GROUND_TRUTH_PARAMS[param_name]
        
        # Calculate mean relative errors
        regime1_rel_errors = calculate_relative_error(regime1_vals, true_val)
        regime2_rel_errors = calculate_relative_error(regime2_vals, true_val)
        
        regime1_mean_rel_err = np.mean(regime1_rel_errors)
        regime2_mean_rel_err = np.mean(regime2_rel_errors)
        
        # Calculate improvement (reduction in relative error)
        if regime1_mean_rel_err > 0:
            improvement = ((regime1_mean_rel_err - regime2_mean_rel_err) / regime1_mean_rel_err * 100)
        else:
            improvement = 0
        
        improvements.append(improvement)
        
        param_display = param_name.replace('_', ' ').title()[:24]
        true_display = f"{true_val:.3f}"
        rel_err1_display = f"{regime1_mean_rel_err:.2f}%"
        rel_err2_display = f"{regime2_mean_rel_err:.2f}%"
        imp_display = f"{improvement:+.1f}%"
        
        row = f"{param_display:<25} {true_display:<12} {rel_err1_display:<15} {rel_err2_display:<15} {imp_display:<12}"
        print(row)
    
    # Overall summary
    if improvements:
        improved_count = sum(1 for x in improvements if x > 0)
        mean_improvement = np.mean(improvements)
        
        print(f"\nOverall Summary:")
        print("-" * 40)
        print(f"Parameters improved in {regime2_name}: {improved_count}/{len(improvements)}")
        print(f"Mean relative error improvement: {mean_improvement:.1f}%")
        
        # Calculate overall relative error statistics
        all_rel_errors_1 = []
        all_rel_errors_2 = []
        
        for param_name in parameter_names:
            regime1_vals = regime1_params.get(param_name, [])
            regime2_vals = regime2_params.get(param_name, [])
            
            if regime1_vals and regime2_vals:
                true_val = GROUND_TRUTH_PARAMS[param_name]
                all_rel_errors_1.extend(calculate_relative_error(regime1_vals, true_val))
                all_rel_errors_2.extend(calculate_relative_error(regime2_vals, true_val))
        
        if all_rel_errors_1 and all_rel_errors_2:
            overall_mean_1 = np.mean(all_rel_errors_1)
            overall_mean_2 = np.mean(all_rel_errors_2)
            overall_median_1 = np.median(all_rel_errors_1)
            overall_median_2 = np.median(all_rel_errors_2)
            
            print(f"\nOverall Relative Error Statistics:")
            print(f"{regime1_name} - Mean: {overall_mean_1:.2f}%, Median: {overall_median_1:.2f}%")
            print(f"{regime2_name} - Mean: {overall_mean_2:.2f}%, Median: {overall_median_2:.2f}%")
        
        if mean_improvement > 5:
            print(f"\n🎉 {regime2_name} training shows significantly better parameter estimation!")
        elif mean_improvement < -5:
            print(f"\n⚠️  {regime1_name} training shows significantly better parameter estimation.")
        else:
            print(f"\n➖ Similar performance between training regimes.")

def save_results_to_csv(regime1_params, regime2_params, regime1_name="1-site", regime2_name="4-site", output_path=None):
    """Save detailed comparison results to CSV files."""
    
    parameter_names = list(GROUND_TRUTH_PARAMS.keys())
    
    # Prepare data for summary CSV
    summary_data = []
    detailed_data = []
    
    for param_name in parameter_names:
        regime1_vals = regime1_params.get(param_name, [])
        regime2_vals = regime2_params.get(param_name, [])
        
        if not regime1_vals or not regime2_vals:
            continue
            
        true_val = GROUND_TRUTH_PARAMS[param_name]
        
        # Calculate relative errors
        regime1_rel_errors = calculate_relative_error(regime1_vals, true_val)
        regime2_rel_errors = calculate_relative_error(regime2_vals, true_val)
        
        # Calculate statistics
        regime1_mean_rel_err = np.mean(regime1_rel_errors)
        regime2_mean_rel_err = np.mean(regime2_rel_errors)
        regime1_median_rel_err = np.median(regime1_rel_errors)
        regime2_median_rel_err = np.median(regime2_rel_errors)
        regime1_std_rel_err = np.std(regime1_rel_errors)
        regime2_std_rel_err = np.std(regime2_rel_errors)
        
        # Calculate improvement
        if regime1_mean_rel_err > 0:
            improvement = ((regime1_mean_rel_err - regime2_mean_rel_err) / regime1_mean_rel_err * 100)
        else:
            improvement = 0
        
        # Add to summary data
        summary_data.append({
            'Parameter': param_name,
            'True_Value': true_val,
            f'{regime1_name}_Mean_RelErr_Percent': regime1_mean_rel_err,
            f'{regime2_name}_Mean_RelErr_Percent': regime2_mean_rel_err,
            f'{regime1_name}_Median_RelErr_Percent': regime1_median_rel_err,
            f'{regime2_name}_Median_RelErr_Percent': regime2_median_rel_err,
            f'{regime1_name}_Std_RelErr_Percent': regime1_std_rel_err,
            f'{regime2_name}_Std_RelErr_Percent': regime2_std_rel_err,
            'Improvement_Percent': improvement,
            f'{regime1_name}_N_Models': len(regime1_vals),
            f'{regime2_name}_N_Models': len(regime2_vals)
        })
        
        # Add to detailed data (individual model results)
        max_models = max(len(regime1_vals), len(regime2_vals))
        for i in range(max_models):
            row = {
                'Parameter': param_name,
                'True_Value': true_val,
                'Model_Index': i
            }
            
            if i < len(regime1_vals):
                row[f'{regime1_name}_Estimated_Value'] = regime1_vals[i]
                row[f'{regime1_name}_RelErr_Percent'] = regime1_rel_errors[i]
            else:
                row[f'{regime1_name}_Estimated_Value'] = np.nan
                row[f'{regime1_name}_RelErr_Percent'] = np.nan
                
            if i < len(regime2_vals):
                row[f'{regime2_name}_Estimated_Value'] = regime2_vals[i]
                row[f'{regime2_name}_RelErr_Percent'] = regime2_rel_errors[i]
            else:
                row[f'{regime2_name}_Estimated_Value'] = np.nan
                row[f'{regime2_name}_RelErr_Percent'] = np.nan
                
            detailed_data.append(row)
    
    # Create DataFrames
    summary_df = pd.DataFrame(summary_data)
    detailed_df = pd.DataFrame(detailed_data)
    
    # Add overall statistics to summary
    if summary_data:
        # Calculate overall statistics
        all_improvements = [row['Improvement_Percent'] for row in summary_data]
        improved_count = sum(1 for x in all_improvements if x > 0)
        mean_improvement = np.mean(all_improvements)
        
        # Add overall row
        overall_row = {
            'Parameter': 'OVERALL_SUMMARY',
            'True_Value': np.nan,
            f'{regime1_name}_Mean_RelErr_Percent': np.nan,
            f'{regime2_name}_Mean_RelErr_Percent': np.nan,
            f'{regime1_name}_Median_RelErr_Percent': np.nan,
            f'{regime2_name}_Median_RelErr_Percent': np.nan,
            f'{regime1_name}_Std_RelErr_Percent': np.nan,
            f'{regime2_name}_Std_RelErr_Percent': np.nan,
            'Improvement_Percent': mean_improvement,
            f'{regime1_name}_N_Models': improved_count,
            f'{regime2_name}_N_Models': len(summary_data)
        }
        
        summary_df = pd.concat([summary_df, pd.DataFrame([overall_row])], ignore_index=True)
    
    # Save to CSV
    if output_path:
        summary_csv_path = f"{output_path}_summary.csv"
        detailed_csv_path = f"{output_path}_detailed.csv"
    else:
        summary_csv_path = "parameter_comparison_summary.csv"
        detailed_csv_path = "parameter_comparison_detailed.csv"
    
    summary_df.to_csv(summary_csv_path, index=False)
    detailed_df.to_csv(detailed_csv_path, index=False)
    
    print(f"\n📊 Results saved to CSV:")
    print(f"   Summary: {summary_csv_path}")
    print(f"   Detailed: {detailed_csv_path}")
    
    return summary_df, detailed_df

def main():
    """Main comparison function."""
    
    # Set paths to your results directories
    regime1_dir = r"C:\NCA_paper\synthetic_results_1site"
    regime2_dir = r"C:\NCA_paper\synthetic_results_4site"
    
    
    
    # Check if directories exist
    if not os.path.exists(regime1_dir):
        print(f"1-site results directory not found: {regime1_dir}")
        return
        
    if not os.path.exists(regime2_dir):
        print(f"4-site results directory not found: {regime2_dir}")
        return
    
    print(f"Comparing parameter estimation between:")
    print(f"- 1-site regime: {regime1_dir}")
    print(f"- 4-site regime: {regime2_dir}")
    
    # Load data from both regimes
    print("\nLoading model data...")
    regime1_params = load_regime_parameters(regime1_dir, "1-site")
    regime2_params = load_regime_parameters(regime2_dir, "4-site")
    
    if not regime1_params or not regime2_params:
        print("Failed to load data from one or both regimes")
        return
    
    # Print simple summary
    print_simple_summary(regime1_params, regime2_params, "1-site", "4-site")
    
    # Create boxplot comparison
    print("\nGenerating comparison visualization...")
    
    # Create output directory
    output_dir = "C:/NCA_paper/dataregime_comparison_results"
    os.makedirs(output_dir, exist_ok=True)
    
    plot_save_path = os.path.join(output_dir, "boxplot")
    create_regime_boxplots(regime1_params, regime2_params, "1-site", "4-site", plot_save_path)
    
    print(f"\n✅ Boxplot saved to: {plot_save_path}_regime_comparison.png")
    save_results_to_csv(regime1_params, regime2_params, "1-site", "4-site", "C:/NCA_paper/dataregime_comparison_results")
if __name__ == "__main__":
    main()
# %%
regime1_dir = r"C:\NCA_paper\synthetic_results_1site"
regime2_dir = r"C:\NCA_paper\synthetic_results_4site"


#%%
# Check if directories exist


print(f"Comparing parameter estimation between:")
print(f"- 1-site regime: {regime1_dir}")
print(f"- 4-site regime: {regime2_dir}")

# Load data from both regimes
print("\nLoading model data...")
regime1_params = load_regime_parameters(regime1_dir, "1-site")
regime2_params = load_regime_parameters(regime2_dir, "4-site")
#%%
def filter_invalid_runs(params_dict, threshold=0.001):
    """
    Filter out runs where any parameter is below the threshold.
    
    Parameters:
    -----------
    params_dict : dict
        Dictionary with parameter names as keys and lists of values as values
    threshold : float
        Minimum valid value for parameters (default: 0.001)
    
    Returns:
    --------
    filtered_dict : dict
        Dictionary with invalid runs removed
    invalid_indices : list
        List of indices that were removed
    """
    # Get number of runs
    n_runs = len(next(iter(params_dict.values())))
    
    # Find invalid runs
    invalid_indices = []
    
    for i in range(n_runs):
        # Check if any parameter is below threshold
        for param_name, values in params_dict.items():
            if values[i] < threshold:
                invalid_indices.append(i)
                print(f"Run {i}: INVALID - {param_name} = {values[i]}")
                break
    
    # Create valid indices mask
    valid_indices = [i for i in range(n_runs) if i not in invalid_indices]
    
    # Filter the dictionary
    filtered_dict = {}
    for param_name, values in params_dict.items():
        filtered_dict[param_name] = [values[i] for i in valid_indices]
    
    print(f"\nTotal runs: {n_runs}")
    print(f"Invalid runs: {len(invalid_indices)}")
    print(f"Valid runs: {len(valid_indices)}")
    print(f"Invalid run indices: {invalid_indices}")
    
    return filtered_dict, invalid_indices

# Apply the filter
regime1_params_filtered, removed_indices = filter_invalid_runs(regime1_params, threshold=0.001)
regime2_params_filtered, removed_indices2 = filter_invalid_runs(regime2_params, threshold=0.001)
# Verify the results
print(f"\nFiltered dictionary has {len(regime1_params_filtered['infiltration_rate'])} runs")
print(f"\nFirst 3 values of each parameter after filtering:")
for param_name, values in regime1_params_filtered.items():
    print(f"{param_name}: {values[:3]}")
#%%
# Print simple summary
print_simple_summary(regime1_params, regime2_params, "1-site", "4-site")

# Create boxplot comparison
print("\nGenerating comparison visualization...")

# Create output directory
output_dir = "C:/NCA_paper/dataregime_comparison_results"
os.makedirs(output_dir, exist_ok=True)

plot_save_path = os.path.join(output_dir, "boxplot")
create_regime_boxplots(regime1_params, regime2_params, "1-site", "4-site", plot_save_path)

print(f"\n✅ Boxplot saved to: {plot_save_path}_regime_comparison.png")
save_results_to_csv(regime1_params_filtered, regime2_params_filtered, "1-site", "4-site", "C:/NCA_paper/dataregime_comparison_results")
# %%

#%%
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import pickle
import warnings
warnings.filterwarnings('ignore')

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

def load_regime_parameters(results_dir: str, regime_name: str):
    """Load final parameters from all models in a regime."""
    
    param_dir = os.path.join(results_dir, "parameters")
    if not os.path.exists(param_dir):
        print(f"Parameters directory not found for {regime_name}: {param_dir}")
        return {}
        
    regime_params = {}
    loaded_models = 0
    
    for model_id in range(30):
        param_file = os.path.join(param_dir, f"model_{model_id:02d}_params.pkl")
        
        if os.path.exists(param_file):
            try:
                with open(param_file, 'rb') as f:
                    parameter_history = pickle.load(f)
                
                if parameter_history:
                    # Get final parameters (last snapshot)
                    final_params = parameter_history[-1].copy()
                    final_params.pop('epoch', None)
                    
                    # Store parameters by name
                    for param_name, param_value in final_params.items():
                        if param_name not in regime_params:
                            regime_params[param_name] = []
                        regime_params[param_name].append(param_value)
                    
                    loaded_models += 1
                    
            except Exception as e:
                print(f"Failed to load {regime_name} model {model_id}: {e}")
                
    print(f"Loaded {loaded_models} models from {regime_name} regime")
    return regime_params

def calculate_relative_error(estimated_values, true_value):
    """Calculate relative error for a set of estimated values."""
    estimated_values = np.array(estimated_values)
    return np.abs(estimated_values - true_value) / np.abs(true_value) * 100

def create_regime_boxplots(regime1_params, regime2_params, regime1_name="One site", regime2_name="Four sites", save_path=None):
    """Create side-by-side boxplots comparing parameter estimates between regimes."""
    
    parameter_names = list(GROUND_TRUTH_PARAMS.keys())
    
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    axes = axes.flatten()
    
    for i, param_name in enumerate(parameter_names):
        ax = axes[i]
        
        # Get parameter values for both regimes
        regime1_vals = regime1_params.get(param_name, [])
        regime2_vals = regime2_params.get(param_name, [])
        
        if not regime1_vals or not regime2_vals:
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{param_name.replace("_", " ").title()}')
            continue
        
        # Calculate relative errors
        true_val = GROUND_TRUTH_PARAMS[param_name]
        regime1_rel_errors = calculate_relative_error(regime1_vals, true_val)
        regime2_rel_errors = calculate_relative_error(regime2_vals, true_val)
        
        # Create boxplots for relative errors
        box_data = [regime1_rel_errors, regime2_rel_errors]
        labels = [regime1_name, regime2_name]
        
        bp = ax.boxplot(box_data, labels=labels, patch_artist=True, widths=0.6, showfliers=False)
        
        # Customize appearance
        colors = ['lightblue', 'lightgreen']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor('none')
            patch.set_edgecolor('black')
        
        # Make median lines more visible
        for median in bp['medians']:
            median.set_color('red')
            median.set_linewidth(2)
        
        # Add individual points (jittered)
        for j, (vals, label) in enumerate(zip(box_data, labels)):
            x_jitter = np.random.normal(j+1, 0.04, size=len(vals))
            ax.scatter(x_jitter, vals, alpha=0.6, s=15, color='darkblue' if j == 0 else 'darkgreen', zorder=3)
        
        ax.set_ylabel('Relative Error (%)')
        ax.set_title(f'{param_name.replace("_", " ").title()}', fontsize=18)
        ax.grid(True, alpha=0.3, axis='y')
        ax.tick_params(axis='x', labelsize=16)
        ax.tick_params(axis='y', labelsize=16)
        
        # Add zero line (perfect estimation)
        ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        
        # Add text showing mean relative errors
        mean1 = np.mean(regime1_rel_errors)
        mean2 = np.mean(regime2_rel_errors)
        stats_text = f'{regime1_name}: {mean1:.1f}%\n{regime2_name}: {mean2:.1f}%'
        ax.text(0.98, 0.98, stats_text, transform=ax.transAxes, 
               verticalalignment='top', horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
               fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(f"{save_path}_regime_comparison.png", dpi=300, bbox_inches='tight')
    
    plt.show()

def print_simple_summary(regime1_params, regime2_params, regime1_name="1-site", regime2_name="4-site"):
    """Print a simple summary of the comparison using relative error."""
    
    print("=" * 80)
    print(f"PARAMETER ESTIMATION COMPARISON: {regime1_name} vs {regime2_name}")
    print("(Using Relative Error)")
    print("=" * 80)
    
    parameter_names = list(GROUND_TRUTH_PARAMS.keys())
    
    print(f"\nParameter-wise Relative Error Comparison:")
    print("-" * 80)
    
    header = f"{'Parameter':<25} {'True Value':<12} {f'{regime1_name} Rel Err':<15} {f'{regime2_name} Rel Err':<15} {'Improvement':<12}"
    print(header)
    print("-" * len(header))
    
    improvements = []
    
    for param_name in parameter_names:
        regime1_vals = regime1_params.get(param_name, [])
        regime2_vals = regime2_params.get(param_name, [])
        
        if not regime1_vals or not regime2_vals:
            continue
            
        true_val = GROUND_TRUTH_PARAMS[param_name]
        
        # Calculate mean relative errors
        regime1_rel_errors = calculate_relative_error(regime1_vals, true_val)
        regime2_rel_errors = calculate_relative_error(regime2_vals, true_val)
        
        regime1_mean_rel_err = np.mean(regime1_rel_errors)
        regime2_mean_rel_err = np.mean(regime2_rel_errors)
        
        # Calculate improvement (reduction in relative error)
        if regime1_mean_rel_err > 0:
            improvement = ((regime1_mean_rel_err - regime2_mean_rel_err) / regime1_mean_rel_err * 100)
        else:
            improvement = 0
        
        improvements.append(improvement)
        
        param_display = param_name.replace('_', ' ').title()[:24]
        true_display = f"{true_val:.3f}"
        rel_err1_display = f"{regime1_mean_rel_err:.2f}%"
        rel_err2_display = f"{regime2_mean_rel_err:.2f}%"
        imp_display = f"{improvement:+.1f}%"
        
        row = f"{param_display:<25} {true_display:<12} {rel_err1_display:<15} {rel_err2_display:<15} {imp_display:<12}"
        print(row)
    
    # Overall summary
    if improvements:
        improved_count = sum(1 for x in improvements if x > 0)
        mean_improvement = np.mean(improvements)
        
        print(f"\nOverall Summary:")
        print("-" * 40)
        print(f"Parameters improved in {regime2_name}: {improved_count}/{len(improvements)}")
        print(f"Mean relative error improvement: {mean_improvement:.1f}%")
        
        # Calculate overall relative error statistics
        all_rel_errors_1 = []
        all_rel_errors_2 = []
        
        for param_name in parameter_names:
            regime1_vals = regime1_params.get(param_name, [])
            regime2_vals = regime2_params.get(param_name, [])
            
            if regime1_vals and regime2_vals:
                true_val = GROUND_TRUTH_PARAMS[param_name]
                all_rel_errors_1.extend(calculate_relative_error(regime1_vals, true_val))
                all_rel_errors_2.extend(calculate_relative_error(regime2_vals, true_val))
        
        if all_rel_errors_1 and all_rel_errors_2:
            overall_mean_1 = np.mean(all_rel_errors_1)
            overall_mean_2 = np.mean(all_rel_errors_2)
            overall_median_1 = np.median(all_rel_errors_1)
            overall_median_2 = np.median(all_rel_errors_2)
            
            print(f"\nOverall Relative Error Statistics:")
            print(f"{regime1_name} - Mean: {overall_mean_1:.2f}%, Median: {overall_median_1:.2f}%")
            print(f"{regime2_name} - Mean: {overall_mean_2:.2f}%, Median: {overall_median_2:.2f}%")
        
        if mean_improvement > 5:
            print(f"\n🎉 {regime2_name} training shows significantly better parameter estimation!")
        elif mean_improvement < -5:
            print(f"\n⚠️  {regime1_name} training shows significantly better parameter estimation.")
        else:
            print(f"\n➖ Similar performance between training regimes.")

def save_results_to_csv(regime1_params, regime2_params, regime1_name="1-site", regime2_name="4-site", output_path=None):
    """Save detailed comparison results to CSV files."""
    
    parameter_names = list(GROUND_TRUTH_PARAMS.keys())
    
    # Prepare data for summary CSV
    summary_data = []
    detailed_data = []
    
    for param_name in parameter_names:
        regime1_vals = regime1_params.get(param_name, [])
        regime2_vals = regime2_params.get(param_name, [])
        
        if not regime1_vals or not regime2_vals:
            continue
            
        true_val = GROUND_TRUTH_PARAMS[param_name]
        
        # Calculate relative errors
        regime1_rel_errors = calculate_relative_error(regime1_vals, true_val)
        regime2_rel_errors = calculate_relative_error(regime2_vals, true_val)
        
        # Calculate statistics
        regime1_mean_rel_err = np.mean(regime1_rel_errors)
        regime2_mean_rel_err = np.mean(regime2_rel_errors)
        regime1_median_rel_err = np.median(regime1_rel_errors)
        regime2_median_rel_err = np.median(regime2_rel_errors)
        regime1_std_rel_err = np.std(regime1_rel_errors)
        regime2_std_rel_err = np.std(regime2_rel_errors)
        
        # Calculate improvement
        if regime1_mean_rel_err > 0:
            improvement = ((regime1_mean_rel_err - regime2_mean_rel_err) / regime1_mean_rel_err * 100)
        else:
            improvement = 0
        
        # Add to summary data
        summary_data.append({
            'Parameter': param_name,
            'True_Value': true_val,
            f'{regime1_name}_Mean_RelErr_Percent': regime1_mean_rel_err,
            f'{regime2_name}_Mean_RelErr_Percent': regime2_mean_rel_err,
            f'{regime1_name}_Median_RelErr_Percent': regime1_median_rel_err,
            f'{regime2_name}_Median_RelErr_Percent': regime2_median_rel_err,
            f'{regime1_name}_Std_RelErr_Percent': regime1_std_rel_err,
            f'{regime2_name}_Std_RelErr_Percent': regime2_std_rel_err,
            'Improvement_Percent': improvement,
            f'{regime1_name}_N_Models': len(regime1_vals),
            f'{regime2_name}_N_Models': len(regime2_vals)
        })
        
        # Add to detailed data (individual model results)
        max_models = max(len(regime1_vals), len(regime2_vals))
        for i in range(max_models):
            row = {
                'Parameter': param_name,
                'True_Value': true_val,
                'Model_Index': i
            }
            
            if i < len(regime1_vals):
                row[f'{regime1_name}_Estimated_Value'] = regime1_vals[i]
                row[f'{regime1_name}_RelErr_Percent'] = regime1_rel_errors[i]
            else:
                row[f'{regime1_name}_Estimated_Value'] = np.nan
                row[f'{regime1_name}_RelErr_Percent'] = np.nan
                
            if i < len(regime2_vals):
                row[f'{regime2_name}_Estimated_Value'] = regime2_vals[i]
                row[f'{regime2_name}_RelErr_Percent'] = regime2_rel_errors[i]
            else:
                row[f'{regime2_name}_Estimated_Value'] = np.nan
                row[f'{regime2_name}_RelErr_Percent'] = np.nan
                
            detailed_data.append(row)
    
    # Create DataFrames
    summary_df = pd.DataFrame(summary_data)
    detailed_df = pd.DataFrame(detailed_data)
    
    # Add overall statistics to summary
    if summary_data:
        # Calculate overall statistics
        all_improvements = [row['Improvement_Percent'] for row in summary_data]
        improved_count = sum(1 for x in all_improvements if x > 0)
        mean_improvement = np.mean(all_improvements)
        
        # Add overall row
        overall_row = {
            'Parameter': 'OVERALL_SUMMARY',
            'True_Value': np.nan,
            f'{regime1_name}_Mean_RelErr_Percent': np.nan,
            f'{regime2_name}_Mean_RelErr_Percent': np.nan,
            f'{regime1_name}_Median_RelErr_Percent': np.nan,
            f'{regime2_name}_Median_RelErr_Percent': np.nan,
            f'{regime1_name}_Std_RelErr_Percent': np.nan,
            f'{regime2_name}_Std_RelErr_Percent': np.nan,
            'Improvement_Percent': mean_improvement,
            f'{regime1_name}_N_Models': improved_count,
            f'{regime2_name}_N_Models': len(summary_data)
        }
        
        summary_df = pd.concat([summary_df, pd.DataFrame([overall_row])], ignore_index=True)
    
    # Save to CSV
    if output_path:
        summary_csv_path = f"{output_path}_summary.csv"
        detailed_csv_path = f"{output_path}_detailed.csv"
    else:
        summary_csv_path = "parameter_comparison_summary.csv"
        detailed_csv_path = "parameter_comparison_detailed.csv"
    
    summary_df.to_csv(summary_csv_path, index=False)
    detailed_df.to_csv(detailed_csv_path, index=False)
    
    print(f"\n📊 Results saved to CSV:")
    print(f"   Summary: {summary_csv_path}")
    print(f"   Detailed: {detailed_csv_path}")
    
    return summary_df, detailed_df

def main():
    """Main comparison function."""
    
    # Set paths to your results directories
    regime1_dir = r"./results/synthetic/synthetic_results_1site"
    regime2_dir = r"./results/synthetic/synthetic_results_4site"
    
    
    
    # Check if directories exist
    if not os.path.exists(regime1_dir):
        print(f"1-site results directory not found: {regime1_dir}")
        return
        
    if not os.path.exists(regime2_dir):
        print(f"4-site results directory not found: {regime2_dir}")
        return
    
    print(f"Comparing parameter estimation between:")
    print(f"- 1-site regime: {regime1_dir}")
    print(f"- 4-site regime: {regime2_dir}")
    
    # Load data from both regimes
    print("\nLoading model data...")
    regime1_params = load_regime_parameters(regime1_dir, "1-site")
    regime2_params = load_regime_parameters(regime2_dir, "4-site")
    
    if not regime1_params or not regime2_params:
        print("Failed to load data from one or both regimes")
        return
    
    # Print simple summary
    print_simple_summary(regime1_params, regime2_params, "1-site", "4-site")
    
    # Create boxplot comparison
    print("\nGenerating comparison visualization...")
    
    # Create output directory
    output_dir = "./results/synthetic/dataregime_comparison_results"
    os.makedirs(output_dir, exist_ok=True)
    
    plot_save_path = os.path.join(output_dir, "boxplot")
    create_regime_boxplots(regime1_params, regime2_params, "1-site", "4-site", plot_save_path)
    
    print(f"\n✅ Boxplot saved to: {plot_save_path}_regime_comparison.png")
    save_results_to_csv(regime1_params, regime2_params, "1-site", "4-site", "C:/NCA_paper/dataregime_comparison_results")
if __name__ == "__main__":
    main()
# %%
regime1_dir = r"./results/synthetic/synthetic_results_1site"
regime2_dir = r"./results/synthetic/synthetic_results_4site"


#%%
# Check if directories exist


print(f"Comparing parameter estimation between:")
print(f"- 1-site regime: {regime1_dir}")
print(f"- 4-site regime: {regime2_dir}")

# Load data from both regimes
print("\nLoading model data...")
regime1_params = load_regime_parameters(regime1_dir, "1-site")
regime2_params = load_regime_parameters(regime2_dir, "4-site")
#%%
def filter_invalid_runs(params_dict, threshold=0.001):
    """
    Filter out runs where any parameter is below the threshold.
    
    Parameters:
    -----------
    params_dict : dict
        Dictionary with parameter names as keys and lists of values as values
    threshold : float
        Minimum valid value for parameters (default: 0.001)
    
    Returns:
    --------
    filtered_dict : dict
        Dictionary with invalid runs removed
    invalid_indices : list
        List of indices that were removed
    """
    # Get number of runs
    n_runs = len(next(iter(params_dict.values())))
    
    # Find invalid runs
    invalid_indices = []
    
    for i in range(n_runs):
        # Check if any parameter is below threshold
        for param_name, values in params_dict.items():
            if values[i] < threshold:
                invalid_indices.append(i)
                print(f"Run {i}: INVALID - {param_name} = {values[i]}")
                break
    
    # Create valid indices mask
    valid_indices = [i for i in range(n_runs) if i not in invalid_indices]
    
    # Filter the dictionary
    filtered_dict = {}
    for param_name, values in params_dict.items():
        filtered_dict[param_name] = [values[i] for i in valid_indices]
    
    print(f"\nTotal runs: {n_runs}")
    print(f"Invalid runs: {len(invalid_indices)}")
    print(f"Valid runs: {len(valid_indices)}")
    print(f"Invalid run indices: {invalid_indices}")
    
    return filtered_dict, invalid_indices

# Apply the filter
regime1_params_filtered, removed_indices = filter_invalid_runs(regime1_params, threshold=0.001)
regime2_params_filtered, removed_indices2 = filter_invalid_runs(regime2_params, threshold=0.001)
# Verify the results
print(f"\nFiltered dictionary has {len(regime1_params_filtered['infiltration_rate'])} runs")
print(f"\nFirst 3 values of each parameter after filtering:")
for param_name, values in regime1_params_filtered.items():
    print(f"{param_name}: {values[:3]}")
#%%
# Print simple summary
print_simple_summary(regime1_params, regime2_params, "1-site", "4-site")

# Create boxplot comparison
print("\nGenerating comparison visualization...")

# Create output directory
output_dir = "./results/synthetic/dataregime_comparison_results"
os.makedirs(output_dir, exist_ok=True)

plot_save_path = os.path.join(output_dir, "boxplot")
create_regime_boxplots(regime1_params, regime2_params, "1-site", "4-site", plot_save_path)

print(f"\n✅ Boxplot saved to: {plot_save_path}_regime_comparison.png")
save_results_to_csv(regime1_params_filtered, regime2_params_filtered, "1-site", "4-site", "C:/NCA_paper/dataregime_comparison_results")

# %%
