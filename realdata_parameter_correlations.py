# -*- coding: utf-8 -*-
"""
Analyzes correlations between learned PDE parameters across training runs
with real satellite observations, to identify trade-offs and couplings
(e.g., does infiltration rate trade off against water use efficiency?).

Produces:
  1. Pairwise scatter matrix of final parameter values
  2. Correlation heatmap (Pearson & Spearman)
  3. Parameter–performance correlation analysis
  4. Training trajectory correlation analysis (do parameters co-evolve?)
"""
#%%
import os
import pickle
import warnings
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings('ignore')

# ── Configuration ──────────────────────────────────────────────────────────
RESULTS_DIR = r"./results/real_data"
OUTPUT_DIR = r"./results/real_data/correlation_analysis"

PARAM_NAMES = [
    'infiltration_rate', 'seepage_rate', 'plant_uptake_rate',
    'mortality_rate', 'evaporation_rate', 'water_use_efficiency',
    'surface_water_diffusion_coeff', 'soil_water_diffusion_coeff',
    'biomass_diffusion_coeff'
]

PARAM_LABELS = {
    'infiltration_rate': 'Infiltration',
    'seepage_rate': 'Seepage',
    'plant_uptake_rate': 'Plant Uptake',
    'mortality_rate': 'Mortality',
    'evaporation_rate': 'Evaporation',
    'water_use_efficiency': 'WUE',
    'surface_water_diffusion_coeff': 'Surf. Water Diff.',
    'soil_water_diffusion_coeff': 'Soil Water Diff.',
    'biomass_diffusion_coeff': 'Biomass Diff.'
}

PERFORMANCE_COLS = ['mse', 'mae', 'correlation']


# ── Data loading ───────────────────────────────────────────────────────────

def load_final_parameters(results_dir: str) -> pd.DataFrame:
    """Load final parameter values from the validation CSV."""
    csv_path = os.path.join(results_dir, "validation_error_analysis.csv")
    df = pd.read_csv(csv_path)
    # Filter out unrealistic runs (has_unrealistic_params == True)
    df_valid = df[df['has_unrealistic_params'] == False].copy()
    df_valid = df_valid.set_index('model_id')
    print(f"Loaded {len(df_valid)} valid models (excluded {len(df) - len(df_valid)} unrealistic)")
    return df_valid


def load_training_trajectories(results_dir: str, model_ids: list) -> dict:
    """Load parameter evolution trajectories from pickle files."""
    param_dir = os.path.join(results_dir, "parameters")
    trajectories = {}
    for mid in model_ids:
        pkl_path = os.path.join(param_dir, f"model_{mid:02d}_params.pkl")
        if os.path.exists(pkl_path):
            with open(pkl_path, 'rb') as f:
                trajectories[mid] = pickle.load(f)
    print(f"Loaded trajectories for {len(trajectories)} models")
    return trajectories


# ── Analysis functions ─────────────────────────────────────────────────────

def compute_correlation_matrices(df: pd.DataFrame, param_names: list):
    """Compute Pearson and Spearman correlation matrices."""
    pearson = df[param_names].corr(method='pearson')
    spearman = df[param_names].corr(method='spearman')
    return pearson, spearman


def compute_pvalues(df: pd.DataFrame, param_names: list):
    """Compute p-values for all pairwise correlations (Spearman)."""
    n = len(param_names)
    pvals = pd.DataFrame(np.ones((n, n)), index=param_names, columns=param_names)
    for i, j in combinations(range(n), 2):
        rho, p = stats.spearmanr(df[param_names[i]], df[param_names[j]])
        pvals.iloc[i, j] = p
        pvals.iloc[j, i] = p
    return pvals


def compute_param_performance_corr(df: pd.DataFrame, param_names: list,
                                    perf_cols: list):
    """Correlate each parameter with performance metrics."""
    rows = []
    for param in param_names:
        for perf in perf_cols:
            rho, p = stats.spearmanr(df[param], df[perf])
            rows.append({'parameter': param, 'metric': perf,
                         'spearman_rho': rho, 'p_value': p})
    return pd.DataFrame(rows)


def compute_trajectory_correlations(trajectories: dict, param_names: list):
    """
    For each model, compute pairwise correlations of parameter changes
    across training epochs. Then average across models.
    This reveals whether parameters systematically co-evolve during training.
    """
    n = len(param_names)
    all_corrs = []

    for mid, history in trajectories.items():
        # Build epoch x param matrix of values
        epochs_data = []
        for snapshot in history:
            if 'epoch' in snapshot and all(p in snapshot for p in param_names):
                epochs_data.append([snapshot[p] for p in param_names])
        if len(epochs_data) < 10:
            continue

        arr = np.array(epochs_data)
        # Use parameter changes (deltas) rather than raw values
        deltas = np.diff(arr, axis=0)
        if deltas.shape[0] < 5:
            continue

        corr_mat = np.corrcoef(deltas.T)
        if not np.any(np.isnan(corr_mat)):
            all_corrs.append(corr_mat)

    if len(all_corrs) == 0:
        return None

    mean_corr = np.mean(all_corrs, axis=0)
    return pd.DataFrame(mean_corr, index=param_names, columns=param_names)


# ── Plotting functions ─────────────────────────────────────────────────────

def plot_correlation_heatmaps(pearson: pd.DataFrame, spearman: pd.DataFrame,
                               pvals: pd.DataFrame, save_path: str):
    """Side-by-side Pearson and Spearman heatmaps with significance markers."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    labels = [PARAM_LABELS.get(c, c) for c in pearson.columns]

    for ax, corr, title in [(ax1, pearson, 'Pearson'), (ax2, spearman, 'Spearman')]:
        im = ax.imshow(corr.values, cmap='RdBu_r', vmin=-1, vmax=1, aspect='equal')
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=10)
        ax.set_yticklabels(labels, fontsize=10)
        ax.set_title(f'{title} Correlation', fontsize=14, fontweight='bold')

        # Annotate cells
        for i in range(len(labels)):
            for j in range(len(labels)):
                val = corr.values[i, j]
                p = pvals.values[i, j]
                sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
                color = 'white' if abs(val) > 0.6 else 'black'
                ax.text(j, i, f'{val:.2f}\n{sig}', ha='center', va='center',
                        fontsize=8, color=color)

    fig.colorbar(im, ax=[ax1, ax2], shrink=0.8, label='Correlation')
    plt.suptitle('Parameter Correlations Across Training Runs\n(* p<0.05, ** p<0.01, *** p<0.001)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'correlation_heatmaps.png'),
                dpi=300, bbox_inches='tight')
    plt.show()


def plot_scatter_matrix(df: pd.DataFrame, param_names: list, save_path: str):
    """Pairwise scatter plots for the most interesting parameter pairs."""
    labels = [PARAM_LABELS.get(c, c) for c in param_names]

    n = len(param_names)
    fig, axes = plt.subplots(n, n, figsize=(2.5 * n, 2.5 * n))

    for i in range(n):
        for j in range(n):
            ax = axes[i, j]
            if i == j:
                ax.hist(df[param_names[i]], bins=10, color='steelblue', alpha=0.7,
                        edgecolor='white')
            else:
                ax.scatter(df[param_names[j]], df[param_names[i]],
                           alpha=0.6, s=25, color='steelblue', edgecolors='white',
                           linewidth=0.5)
                # Add trend line
                x, y = df[param_names[j]].values, df[param_names[i]].values
                mask = np.isfinite(x) & np.isfinite(y)
                if mask.sum() > 3:
                    z = np.polyfit(x[mask], y[mask], 1)
                    xline = np.linspace(x[mask].min(), x[mask].max(), 50)
                    ax.plot(xline, np.polyval(z, xline), 'r-', alpha=0.5, linewidth=1)

            if j == 0:
                ax.set_ylabel(labels[i], fontsize=8)
            else:
                ax.set_yticklabels([])
            if i == n - 1:
                ax.set_xlabel(labels[j], fontsize=8)
            else:
                ax.set_xticklabels([])
            ax.tick_params(labelsize=6)

    plt.suptitle('Pairwise Parameter Scatter Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'scatter_matrix.png'),
                dpi=200, bbox_inches='tight')
    plt.show()


def plot_param_performance(corr_df: pd.DataFrame, save_path: str):
    """Bar chart of parameter–performance Spearman correlations."""
    fig, axes = plt.subplots(1, len(PERFORMANCE_COLS), figsize=(6 * len(PERFORMANCE_COLS), 6))
    if len(PERFORMANCE_COLS) == 1:
        axes = [axes]

    for ax, metric in zip(axes, PERFORMANCE_COLS):
        sub = corr_df[corr_df['metric'] == metric].copy()
        sub['label'] = sub['parameter'].map(PARAM_LABELS)
        sub = sub.sort_values('spearman_rho')

        colors = ['firebrick' if p < 0.05 else 'lightgray'
                  for p in sub['p_value']]
        ax.barh(sub['label'], sub['spearman_rho'], color=colors, edgecolor='white')
        ax.axvline(0, color='black', linewidth=0.5)
        ax.set_xlabel('Spearman rho')
        ax.set_title(f'Correlation with {metric.upper()}', fontweight='bold')
        ax.set_xlim(-1, 1)

    plt.suptitle('Parameter–Performance Correlations\n(red = p < 0.05)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'param_performance_corr.png'),
                dpi=300, bbox_inches='tight')
    plt.show()


def plot_trajectory_correlations(traj_corr: pd.DataFrame, save_path: str):
    """Heatmap of mean pairwise correlations of parameter *changes* during training."""
    labels = [PARAM_LABELS.get(c, c) for c in traj_corr.columns]

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(traj_corr.values, cmap='RdBu_r', vmin=-1, vmax=1, aspect='equal')
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=10)
    ax.set_yticklabels(labels, fontsize=10)

    for i in range(len(labels)):
        for j in range(len(labels)):
            val = traj_corr.values[i, j]
            color = 'white' if abs(val) > 0.6 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                    fontsize=9, color=color)

    fig.colorbar(im, ax=ax, shrink=0.8, label='Mean correlation of epoch-to-epoch changes')
    ax.set_title('Training Trajectory Co-evolution\n'
                 '(correlation of parameter changes across epochs, averaged over models)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'trajectory_coevolution.png'),
                dpi=300, bbox_inches='tight')
    plt.show()


def print_top_correlations(spearman: pd.DataFrame, pvals: pd.DataFrame, n_top: int = 10):
    """Print the strongest pairwise correlations."""
    pairs = []
    params = spearman.columns.tolist()
    for i, j in combinations(range(len(params)), 2):
        pairs.append({
            'param_1': PARAM_LABELS.get(params[i], params[i]),
            'param_2': PARAM_LABELS.get(params[j], params[j]),
            'spearman_rho': spearman.iloc[i, j],
            'p_value': pvals.iloc[i, j]
        })
    pairs_df = pd.DataFrame(pairs).sort_values('spearman_rho', key=abs, ascending=False)

    print(f"\n{'='*70}")
    print(f"TOP {n_top} STRONGEST PARAMETER CORRELATIONS (by |Spearman rho|)")
    print(f"{'='*70}")
    for _, row in pairs_df.head(n_top).iterrows():
        sig = '***' if row['p_value'] < 0.001 else '**' if row['p_value'] < 0.01 \
              else '*' if row['p_value'] < 0.05 else 'n.s.'
        direction = 'TRADE-OFF' if row['spearman_rho'] < 0 else 'COUPLING'
        print(f"  {row['param_1']:>20s}  vs  {row['param_2']:<20s}  "
              f"rho={row['spearman_rho']:+.3f}  p={row['p_value']:.4f}  "
              f"({sig})  [{direction}]")

    return pairs_df


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Load data
    df = load_final_parameters(RESULTS_DIR)

    # 2. Correlation matrices on final parameter values
    pearson, spearman = compute_correlation_matrices(df, PARAM_NAMES)
    pvals = compute_pvalues(df, PARAM_NAMES)

    # 3. Print strongest correlations
    pairs_df = print_top_correlations(spearman, pvals)
    pairs_df.to_csv(os.path.join(OUTPUT_DIR, 'pairwise_correlations.csv'), index=False)

    # 4. Plots
    plot_correlation_heatmaps(pearson, spearman, pvals, OUTPUT_DIR)
    plot_scatter_matrix(df, PARAM_NAMES, OUTPUT_DIR)

    # 5. Parameter–performance correlations
    perf_corr = compute_param_performance_corr(df, PARAM_NAMES, PERFORMANCE_COLS)
    print(f"\n{'='*70}")
    print("PARAMETER–PERFORMANCE CORRELATIONS")
    print(f"{'='*70}")
    for metric in PERFORMANCE_COLS:
        sub = perf_corr[perf_corr['metric'] == metric].sort_values('spearman_rho', key=abs, ascending=False)
        print(f"\n  {metric.upper()}:")
        for _, row in sub.iterrows():
            sig = '*' if row['p_value'] < 0.05 else ''
            print(f"    {PARAM_LABELS[row['parameter']]:>20s}  rho={row['spearman_rho']:+.3f}  "
                  f"p={row['p_value']:.4f} {sig}")
    perf_corr.to_csv(os.path.join(OUTPUT_DIR, 'param_performance_correlations.csv'), index=False)
    plot_param_performance(perf_corr, OUTPUT_DIR)

    # 6. Training trajectory co-evolution
    print("\nAnalyzing training trajectory co-evolution...")
    trajectories = load_training_trajectories(RESULTS_DIR, df.index.tolist())
    traj_corr = compute_trajectory_correlations(trajectories, PARAM_NAMES)
    if traj_corr is not None:
        plot_trajectory_correlations(traj_corr, OUTPUT_DIR)
        traj_corr.to_csv(os.path.join(OUTPUT_DIR, 'trajectory_coevolution.csv'))
        print("Trajectory co-evolution matrix saved.")
    else:
        print("Could not compute trajectory correlations (insufficient data).")

    print(f"\nAll outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

# %%
