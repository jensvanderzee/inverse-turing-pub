# -*- coding: utf-8 -*-
"""
Test retrained PDE models on held-out sites (f, k, j).
Models were trained on sites (b, i, c, e) using realdata_train_invPDE.py.
Parameters loaded from four_site_final_parameter_values.csv.
"""
#%%
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import json

from realdata_train_invPDE import invRietkerk, RealDataLoader, EcologicalParameters

# ── Configuration ───────────────────────────────────────────────────────────
PARAM_CSV = "results/parameter_history_analysis/four_site_final_parameter_values.csv"
DATA_DIR = "data"
TEST_SITES = ["f", "k", "j"]
SAVE_DIR = "results/real_data/test_results"
NDVI_TO_BIOMASS_MULTIPLIER = 1500.0
STEPS_PER_WEEK = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PARAM_NAMES = [
    "surface_water_diffusion_coeff",
    "soil_water_diffusion_coeff",
    "biomass_diffusion_coeff",
    "evaporation_rate",
    "seepage_rate",
    "mortality_rate",
    "infiltration_rate",
    "plant_uptake_rate",
    "water_use_efficiency",
]


# ── Helpers ─────────────────────────────────────────────────────────────────
def load_parameters(csv_path: str) -> pd.DataFrame:
    """Load the parameter CSV. Index column is the model ID."""
    df = pd.read_csv(csv_path, index_col=0)
    return df


def build_model_from_row(row: pd.Series) -> invRietkerk:
    """Create an invRietkerk model and set its parameters from a CSV row."""
    model = invRietkerk(trainable=True).to(DEVICE)
    with torch.no_grad():
        for name in PARAM_NAMES:
            param = getattr(model, name)
            param.data.fill_(row[name])
    return model


def evaluate_model_on_site(
    model: invRietkerk,
    time_series: list,
) -> dict:
    """
    Run forward simulation on a single test site and compute metrics.

    Mirrors the training loss: delta-based MSE over consecutive year pairs.
    Also computes MAE on deltas and pixel-wise correlation on final biomass.
    """
    loss_fn = nn.MSELoss()
    mae_fn = nn.L1Loss()

    initial_biomass = time_series[0]["biomass"].clone().to(DEVICE)
    pred_surface_water = torch.zeros_like(initial_biomass, device=DEVICE)
    pred_soil_water = torch.zeros_like(initial_biomass, device=DEVICE)
    pred_biomass = initial_biomass.clone()

    total_mse = 0.0
    total_mae = 0.0
    correlations = []
    num_transitions = 0

    with torch.no_grad():
        for t_idx in range(len(time_series) - 1):
            observed_current = time_series[t_idx]["biomass"].to(DEVICE)
            observed_next = time_series[t_idx + 1]["biomass"].to(DEVICE)
            observed_delta = observed_next - observed_current

            initial_pred_biomass = pred_biomass.clone()

            weekly_precip = time_series[t_idx]["weekly_precipitation"]
            pred_surface_water, pred_soil_water, pred_biomass = (
                model.simulate_year_weekly(
                    pred_surface_water,
                    pred_soil_water,
                    pred_biomass,
                    weekly_precipitation=weekly_precip,
                    steps_per_week=STEPS_PER_WEEK,
                )
            )

            predicted_delta = pred_biomass - initial_pred_biomass

            mse = loss_fn(predicted_delta, observed_delta).item()
            mae = mae_fn(predicted_delta, observed_delta).item()
            total_mse += mse
            total_mae += mae

            # Pixel-wise correlation between predicted and observed final biomass
            pred_flat = pred_biomass.flatten().cpu().numpy()
            obs_flat = observed_next.flatten().cpu().numpy()
            if np.std(pred_flat) > 0 and np.std(obs_flat) > 0:
                corr = np.corrcoef(pred_flat, obs_flat)[0, 1]
            else:
                corr = 0.0
            correlations.append(corr)
            num_transitions += 1

    if num_transitions > 0:
        avg_mse = total_mse / num_transitions
        avg_mae = total_mae / num_transitions
        avg_corr = np.mean(correlations)
    else:
        avg_mse = avg_mae = avg_corr = float("nan")

    return {
        "mse": avg_mse,
        "mae": avg_mae,
        "correlation": avg_corr,
        "num_transitions": num_transitions,
    }


# ── Main ────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    # 1. Load parameters
    print("Loading model parameters...")
    param_df = load_parameters(PARAM_CSV)
    print(f"  Loaded {len(param_df)} models from {PARAM_CSV}")

    # 2. Load test site data
    print(f"\nLoading test site data for sites {TEST_SITES}...")
    data_loader = RealDataLoader(
        DATA_DIR, selected_sites=TEST_SITES, device=DEVICE, use_weekly_precip=True
    )
    test_data = data_loader.get_training_data(
        ndvi_to_biomass_multiplier=NDVI_TO_BIOMASS_MULTIPLIER
    )

    print(f"\nTest sites loaded: {list(test_data['location_time_series'].keys())}")
    for loc, ts in test_data["location_time_series"].items():
        years = [t["year"] for t in ts]
        print(f"  {loc}: {len(ts)} time points, years {min(years)}-{max(years)}")

    # 3. Evaluate each model on each test site
    print(f"\nEvaluating {len(param_df)} models on {len(test_data['location_time_series'])} test sites...")
    results = []

    for model_id, row in param_df.iterrows():
        model = build_model_from_row(row)
        model.eval()

        for site_name, time_series in test_data["location_time_series"].items():
            metrics = evaluate_model_on_site(model, time_series)
            results.append(
                {
                    "model_id": model_id,
                    "site": site_name,
                    "mse": metrics["mse"],
                    "mae": metrics["mae"],
                    "correlation": metrics["correlation"],
                    "num_transitions": metrics["num_transitions"],
                    "turing_value": row["turing_value"],
                }
            )

        print(f"  Model {model_id} done")

    # 4. Save per-model per-site results
    results_df = pd.DataFrame(results)
    results_csv = os.path.join(SAVE_DIR, "test_metrics.csv")
    results_df.to_csv(results_csv, index=False)
    print(f"\nPer-model metrics saved to {results_csv}")

    # 5. Summary statistics
    print("\n" + "=" * 60)
    print("TEST RESULTS SUMMARY")
    print("=" * 60)

    summary = {}
    for site_name in results_df["site"].unique():
        site_df = results_df[results_df["site"] == site_name]
        stats = {
            "mse_mean": float(site_df["mse"].mean()),
            "mse_std": float(site_df["mse"].std()),
            "mse_min": float(site_df["mse"].min()),
            "mse_max": float(site_df["mse"].max()),
            "mae_mean": float(site_df["mae"].mean()),
            "mae_std": float(site_df["mae"].std()),
            "corr_mean": float(site_df["correlation"].mean()),
            "corr_std": float(site_df["correlation"].std()),
            "num_models": int(len(site_df)),
        }
        summary[site_name] = stats
        print(f"\n{site_name}:")
        print(f"  MSE:  {stats['mse_mean']:.2f} +/- {stats['mse_std']:.2f}  (min={stats['mse_min']:.2f}, max={stats['mse_max']:.2f})")
        print(f"  MAE:  {stats['mae_mean']:.2f} +/- {stats['mae_std']:.2f}")
        print(f"  Corr: {stats['corr_mean']:.4f} +/- {stats['corr_std']:.4f}")

    # Overall across all sites
    overall = {
        "mse_mean": float(results_df["mse"].mean()),
        "mae_mean": float(results_df["mae"].mean()),
        "corr_mean": float(results_df["correlation"].mean()),
    }
    summary["overall"] = overall
    print(f"\nOverall (all sites):")
    print(f"  MSE:  {overall['mse_mean']:.2f}")
    print(f"  MAE:  {overall['mae_mean']:.2f}")
    print(f"  Corr: {overall['corr_mean']:.4f}")

    # Save summary JSON
    summary_path = os.path.join(SAVE_DIR, "test_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {summary_path}")

    # 6. Visualizations
    _plot_metrics_boxplot(results_df)
    _plot_best_model_predictions(results_df, param_df, test_data)

    print(f"\nAll outputs saved to {SAVE_DIR}/")


def _plot_metrics_boxplot(results_df: pd.DataFrame):
    """Box plots of MSE, MAE, and correlation per test site."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    sites = sorted(results_df["site"].unique())

    for ax, metric, title in zip(
        axes, ["mse", "mae", "correlation"], ["MSE", "MAE", "Correlation"]
    ):
        data = [results_df[results_df["site"] == s][metric].values for s in sites]
        ax.boxplot(data, labels=sites)
        ax.set_title(title)
        ax.set_xlabel("Test site")
        ax.set_ylabel(title)

    fig.suptitle("Test Metrics Across Models per Site", fontsize=14)
    plt.tight_layout()
    fig.savefig(os.path.join(SAVE_DIR, "test_metrics_boxplot.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved test_metrics_boxplot.png")


def _plot_best_model_predictions(
    results_df: pd.DataFrame,
    param_df: pd.DataFrame,
    test_data: dict,
):
    """For the best model (lowest mean test MSE), plot predicted vs observed biomass."""
    # Find best model by mean MSE across all test sites
    mean_mse = results_df.groupby("model_id")["mse"].mean()
    best_model_id = mean_mse.idxmin()
    print(f"\n  Best model by mean test MSE: model {best_model_id} (MSE={mean_mse[best_model_id]:.2f})")

    model = build_model_from_row(param_df.loc[best_model_id])
    model.eval()

    for site_name, time_series in test_data["location_time_series"].items():
        n_transitions = len(time_series) - 1
        if n_transitions == 0:
            continue

        # Run forward simulation
        initial_biomass = time_series[0]["biomass"].clone().to(DEVICE)
        pred_sw = torch.zeros_like(initial_biomass, device=DEVICE)
        pred_gw = torch.zeros_like(initial_biomass, device=DEVICE)
        pred_b = initial_biomass.clone()

        predicted_biomass_list = [initial_biomass.squeeze().cpu().numpy()]
        observed_biomass_list = [initial_biomass.squeeze().cpu().numpy()]

        with torch.no_grad():
            for t_idx in range(n_transitions):
                weekly_precip = time_series[t_idx]["weekly_precipitation"]
                pred_sw, pred_gw, pred_b = model.simulate_year_weekly(
                    pred_sw, pred_gw, pred_b,
                    weekly_precipitation=weekly_precip,
                    steps_per_week=STEPS_PER_WEEK,
                )
                predicted_biomass_list.append(pred_b.squeeze().cpu().numpy())
                observed_biomass_list.append(
                    time_series[t_idx + 1]["biomass"].squeeze().cpu().numpy()
                )

        # Plot: rows = years, cols = [observed, predicted, difference]
        n_years = len(predicted_biomass_list)
        fig, axes = plt.subplots(n_years, 3, figsize=(12, 3.5 * n_years))
        if n_years == 1:
            axes = axes[np.newaxis, :]

        vmax = NDVI_TO_BIOMASS_MULTIPLIER * 0.5  # reasonable upper bound for display

        for i in range(n_years):
            year = time_series[i]["year"]
            obs = observed_biomass_list[i]
            pred = predicted_biomass_list[i]
            diff = pred - obs

            im0 = axes[i, 0].imshow(obs, cmap="RdYlGn", vmin=0, vmax=vmax)
            axes[i, 0].set_title(f"Observed {year}")
            axes[i, 0].axis("off")
            plt.colorbar(im0, ax=axes[i, 0], fraction=0.046, pad=0.04)

            im1 = axes[i, 1].imshow(pred, cmap="RdYlGn", vmin=0, vmax=vmax)
            axes[i, 1].set_title(f"Predicted {year}")
            axes[i, 1].axis("off")
            plt.colorbar(im1, ax=axes[i, 1], fraction=0.046, pad=0.04)

            abs_max = max(abs(diff.min()), abs(diff.max()), 1e-6)
            im2 = axes[i, 2].imshow(diff, cmap="RdBu_r", vmin=-abs_max, vmax=abs_max)
            axes[i, 2].set_title(f"Difference {year}")
            axes[i, 2].axis("off")
            plt.colorbar(im2, ax=axes[i, 2], fraction=0.046, pad=0.04)

        fig.suptitle(
            f"Best Model ({best_model_id}) — {site_name}",
            fontsize=14,
        )
        plt.tight_layout()
        fig.savefig(
            os.path.join(SAVE_DIR, f"best_model_predictions_{site_name}.png"),
            dpi=150,
            bbox_inches="tight",
        )
        plt.close(fig)
        print(f"  Saved best_model_predictions_{site_name}.png")


if __name__ == "__main__":
    main()

# %%
