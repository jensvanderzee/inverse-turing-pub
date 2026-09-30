#%%
import os
import pickle
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ════════════════════════════════════════════════════════════════════════════
# SETTINGS
# ════════════════════════════════════════════════════════════════════════════
BASE_DIR = "results/real_data/models_multiplier_check"
SAVE_DIR = os.path.join(BASE_DIR, "results")

# Multipliers and their corresponding file stems
MULTIPLIERS = {
    750:  {"param_pkl": "model_750_00_params.pkl",  "model_pth": "model_750_00.pth"},
    1000: {"param_pkl": "model_1000_00_params.pkl", "model_pth": "model_1000_00.pth"},
    1500: {"param_pkl": "model_1500_00_params.pkl", "model_pth": "model_1500_00.pth"},
    2250: {"param_pkl": "model_2250_00_params.pkl", "model_pth": "model_2250_00.pth"},
    3000: {"param_pkl": "model_3000_00_params.pkl", "model_pth": "model_3000_00.pth"},
}

COLOURS = {750: "#1b9e77", 1000: "#d95f02", 1500: "#7570b3",
           2250: "#e7298a", 3000: "#66a61e"}
# ════════════════════════════════════════════════════════════════════════════

os.makedirs(SAVE_DIR, exist_ok=True)


# ── 1. Load parameter histories ───────────────────────────────────────────
param_histories = {}
for mult, files in MULTIPLIERS.items():
    pkl_path = os.path.join(BASE_DIR, "parameters", files["param_pkl"])
    with open(pkl_path, "rb") as f:
        history = pickle.load(f)
    param_histories[mult] = history
    print(f"Multiplier {mult:>5d}: {len(history)} snapshots, "
          f"epochs {history[0]['epoch']}–{history[-1]['epoch']}")

# Identify parameter names (everything except 'epoch')
first_mult = next(iter(param_histories))
param_names = [k for k in param_histories[first_mult][0].keys() if k != "epoch"]
print(f"\nParameters ({len(param_names)}): {param_names}")


# ── 2. Load loss histories from .pth checkpoints (if available) ───────────
loss_histories = {}
for mult, files in MULTIPLIERS.items():
    pth_path = os.path.join(BASE_DIR, "models", files["model_pth"])
    ckpt = torch.load(pth_path, map_location="cpu", weights_only=False)
    if "loss_history" in ckpt:
        loss_histories[mult] = ckpt["loss_history"]
        print(f"Multiplier {mult:>5d}: {len(ckpt['loss_history'])} loss values, "
              f"final loss = {ckpt['loss_history'][-1]:.4f}")
    elif "losses" in ckpt:
        loss_histories[mult] = ckpt["losses"]
        print(f"Multiplier {mult:>5d}: {len(ckpt['losses'])} loss values, "
              f"final loss = {ckpt['losses'][-1]:.4f}")
    else:
        print(f"Multiplier {mult:>5d}: no loss history found in checkpoint "
              f"(keys: {list(ckpt.keys())})")


# ── 3. Build DataFrames for easy comparison ───────────────────────────────
# Final parameter values
final_rows = []
for mult, history in param_histories.items():
    row = {"multiplier": mult}
    row.update({p: history[-1][p] for p in param_names})
    final_rows.append(row)
final_df = pd.DataFrame(final_rows).set_index("multiplier")

print("\n" + "=" * 70)
print("FINAL PARAMETER VALUES")
print("=" * 70)
print(final_df.to_string(float_format="{:.6f}".format))

# Initial parameter values (for reference)
init_rows = []
for mult, history in param_histories.items():
    row = {"multiplier": mult}
    row.update({p: history[0][p] for p in param_names})
    init_rows.append(row)
init_df = pd.DataFrame(init_rows).set_index("multiplier")

# Parameter displacement per multiplier
disp_rows = []
for mult in MULTIPLIERS:
    row = {"multiplier": mult}
    for p in param_names:
        row[p] = abs(param_histories[mult][-1][p] - param_histories[mult][0][p])
    disp_rows.append(row)
disp_df = pd.DataFrame(disp_rows).set_index("multiplier")
disp_df["total_displacement"] = disp_df.sum(axis=1)

print("\n" + "=" * 70)
print("PARAMETER DISPLACEMENT (|final - initial|)")
print("=" * 70)
print(disp_df.to_string(float_format="{:.6f}".format))

# Save summary
final_df.to_csv(os.path.join(SAVE_DIR, "multiplier_final_params.csv"))
disp_df.to_csv(os.path.join(SAVE_DIR, "multiplier_param_displacement.csv"))
print(f"\nSaved CSVs to {SAVE_DIR}/")


# ── 3b. Rainfall→biomass conversion efficiency vs multiplier ─────────────
#    (computed BEFORE cosmetic scaling so it uses original parameter values)
NDVI_AVG = 0.15
mults = sorted(MULTIPLIERS.keys())

def _compute_efficiency(params, multiplier, ndvi=NDVI_AVG):
    I = params["infiltration_rate"]
    E = params["evaporation_rate"]
    WUE = params["water_use_efficiency"]
    U = params["plant_uptake_rate"]
    S = params["seepage_rate"]
    infil_term = (I * multiplier * ndvi) / (I * multiplier * ndvi + E)
    uptake_term = (U * multiplier * ndvi) / (U * multiplier * ndvi + S)
    return infil_term * WUE * uptake_term

required = {"infiltration_rate", "evaporation_rate", "water_use_efficiency",
            "plant_uptake_rate", "seepage_rate"}
if required.issubset(set(param_names)):
    eff_rows = []
    for mult in mults:
        final_params = param_histories[mult][-1]
        eff = _compute_efficiency(final_params, mult)
        eff_rows.append({"multiplier": mult, "efficiency": eff})
    eff_df = pd.DataFrame(eff_rows).set_index("multiplier")
    eff_df.to_csv(os.path.join(SAVE_DIR, "multiplier_efficiency.csv"))

    print("\n" + "=" * 70)
    print(f"RAINFALL→BIOMASS EFFICIENCY (NDVI={NDVI_AVG})")
    print("=" * 70)
    print(eff_df.to_string(float_format="{:.6f}".format))
else:
    missing = required - set(param_names)
    print(f"\nSkipping efficiency calc — missing parameters: {missing}")
    eff_df = None


# ── 3c. Cosmetic scaling for DISPLAY ONLY ─────────────────────────────────
SCALE_FACTOR = 900
PARAMS_TO_SCALE = [p for p in param_names
                   if "diffusion" in p or p in ("infiltration_rate", "plant_uptake_rate")]

print(f"\nScaling {PARAMS_TO_SCALE} by {SCALE_FACTOR} (display only)")

for mult in param_histories:
    for snapshot in param_histories[mult]:
        for p in PARAMS_TO_SCALE:
            snapshot[p] *= SCALE_FACTOR

for p in PARAMS_TO_SCALE:
    final_df[p]  *= SCALE_FACTOR
    init_df[p]   *= SCALE_FACTOR
    disp_df[p]   *= SCALE_FACTOR

disp_df["total_displacement"] = disp_df[param_names].sum(axis=1)


#%%
# ── 4. Plot: Parameter training trajectories ──────────────────────────────
ncols = 3
nrows = (len(param_names) + ncols - 1) // ncols
fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 4 * nrows),
                         sharex=True)
axes_flat = axes.flatten()

for i, pname in enumerate(param_names):
    ax = axes_flat[i]
    for mult, history in param_histories.items():
        epochs = [s["epoch"] for s in history]
        values = [s[pname] for s in history]
        ax.plot(epochs, values, color=COLOURS[mult], label=f"{mult}", linewidth=1.2)
    ax.set_title(pname.replace("_", " ").capitalize(), fontsize=14)
    ax.set_ylabel("value", fontsize=14)
    ax.grid(True, alpha=0.3)
    if i == 2:
        ax.legend(title="NDVI multiplier", fontsize=9)
    ax.tick_params(axis='both', which='major', labelsize=13)
# Hide unused axes
for j in range(len(param_names), len(axes_flat)):
    axes_flat[j].axis("off")

for ax in axes_flat[max(0, nrows * ncols - ncols):]:
    ax.set_xlabel("Epoch", fontsize=14)

fig.suptitle("", fontsize=14, y=1.01)
plt.tight_layout()
fig.savefig(os.path.join(SAVE_DIR, "multiplier_param_trajectories.png"),
            dpi=150, bbox_inches="tight")
plt.show()
plt.close(fig)
print("Saved multiplier_param_trajectories.png")

#%%
# ── 5. Plot: Final parameter comparison (grouped bar chart) ──────────────
x = np.arange(len(param_names))
width = 0.16

fig, ax = plt.subplots(figsize=(16, 5))
for i, mult in enumerate(mults):
    vals = [final_df.loc[mult, p] for p in param_names]
    ax.bar(x + i * width, vals, width, label=f"{mult}", color=COLOURS[mult])

ax.set_xticks(x + 2 * width)
ax.set_xticklabels([p.replace("_", "\n") for p in param_names], fontsize=9)
ax.set_ylabel("Final parameter value")
ax.set_title("Final parameter values by NDVI-to-biomass multiplier")
ax.legend(title="Multiplier")
ax.grid(True, alpha=0.3, axis="y")
plt.tight_layout()
fig.savefig(os.path.join(SAVE_DIR, "multiplier_final_params_bar.png"),
            dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved multiplier_final_params_bar.png")


# ── 6. Plot: Final parameters on log scale ────────────────────────────────
fig, ax = plt.subplots(figsize=(16, 5))
for i, mult in enumerate(mults):
    vals = [final_df.loc[mult, p] for p in param_names]
    ax.bar(x + i * width, vals, width, label=f"{mult}", color=COLOURS[mult])

ax.set_yscale("log")
ax.set_xticks(x + 2 * width)
ax.set_xticklabels([p.replace("_", "\n") for p in param_names], fontsize=9)
ax.set_ylabel("Final parameter value (log scale)")
ax.set_title("Final parameter values by NDVI-to-biomass multiplier (log scale)")
ax.legend(title="Multiplier")
ax.grid(True, alpha=0.3, axis="y")
plt.tight_layout()
fig.savefig(os.path.join(SAVE_DIR, "multiplier_final_params_bar_log.png"),
            dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved multiplier_final_params_bar_log.png")


# ── 7. Plot: Loss trajectories ────────────────────────────────────────────
if loss_histories:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Linear scale
    ax = axes[0]
    for mult, losses in sorted(loss_histories.items()):
        ax.plot(losses, color=COLOURS[mult], label=f"{mult}", linewidth=1.0)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training loss (linear scale)")
    ax.legend(title="Multiplier")
    ax.grid(True, alpha=0.3)

    # Log scale
    ax = axes[1]
    for mult, losses in sorted(loss_histories.items()):
        ax.plot(losses, color=COLOURS[mult], label=f"{mult}", linewidth=1.0)
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (log scale)")
    ax.set_title("Training loss (log scale)")
    ax.legend(title="Multiplier")
    ax.grid(True, alpha=0.3)

    plt.suptitle("Training loss by NDVI-to-biomass multiplier", fontsize=14)
    plt.tight_layout()
    fig.savefig(os.path.join(SAVE_DIR, "multiplier_loss_trajectories.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved multiplier_loss_trajectories.png")
else:
    print("No loss histories found in checkpoints — skipping loss plot.")


# ── 8. Plot: Parameter displacement comparison ───────────────────────────
fig, ax = plt.subplots(figsize=(16, 5))
for i, mult in enumerate(mults):
    vals = [disp_df.loc[mult, p] for p in param_names]
    ax.bar(x + i * width, vals, width, label=f"{mult}", color=COLOURS[mult])

ax.set_xticks(x + 2 * width)
ax.set_xticklabels([p.replace("_", "\n") for p in param_names], fontsize=9)
ax.set_ylabel("|final - initial|")
ax.set_title("Parameter displacement by NDVI-to-biomass multiplier")
ax.legend(title="Multiplier")
ax.grid(True, alpha=0.3, axis="y")
plt.tight_layout()
fig.savefig(os.path.join(SAVE_DIR, "multiplier_param_displacement.png"),
            dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved multiplier_param_displacement.png")


#%%
# ── 8b. Plot: Efficiency vs multiplier ────────────────────────────────────
if eff_df is not None:
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot(eff_df.index, eff_df["efficiency"], marker="o", linewidth=1.5,
            color="#333333")
    for mult in mults:
        ax.scatter(mult, eff_df.loc[mult, "efficiency"],
                   color="black", s=80, zorder=5, label=f"{mult}")
    ax.set_xlabel("NDVI-to-biomass multiplier ($\\lambda$)", fontsize=17)
    ax.set_ylabel("Rain use efficiency", fontsize=17)
    ax.set_title("")
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.set_ylim((0, 0.15))
    plt.tight_layout()
    fig.savefig(os.path.join(SAVE_DIR, "multiplier_efficiency.png"),
                dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    print("Saved multiplier_efficiency.png")


#%%
# ── 9. Summary print ─────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)
print(f"Outputs saved to: {SAVE_DIR}/")
print(f"  - multiplier_param_trajectories.png   (training histories)")
print(f"  - multiplier_final_params_bar.png      (final values, linear)")
print(f"  - multiplier_final_params_bar_log.png  (final values, log)")
if loss_histories:
    print(f"  - multiplier_loss_trajectories.png     (loss curves)")
print(f"  - multiplier_param_displacement.png    (displacement)")
print(f"  - multiplier_final_params.csv          (final values table)")
print(f"  - multiplier_param_displacement.csv    (displacement table)")
if eff_df is not None:
    print(f"  - multiplier_efficiency.png            (efficiency vs multiplier)")
    print(f"  - multiplier_efficiency.csv            (efficiency table)")

# %%