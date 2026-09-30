"""Select the best RCNN model from `synthetic_rcnn_4site_retrain` and the
best inverse-PDE model from `synthetic_invPDE_4site` by training loss,
then test how they extrapolate to (a) longer time horizons and
(b) precipitation regimes outside the 4 training sites, using the
ground-truth PDE (`invRietkerk`) as reference.

Reuses:
  - train_rcnn_batch.invRietkerk              (PDE ground truth, fixed params)
  - train_rcnn_batch.SyntheticDataGenerator   (equilibrium initial state)
  - train_rcnn_batch.generate_weekly_precipitation
  - train_rcnn_batch.RCNNBaseline             (model class)
  - train_invPDE_synthetic_batch.invRietkerk  (inverse PDE with learned params)

Run from the repo root:
    python rcnn_extrapolation_test.py
"""
#%%
import glob
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from train_rcnn_batch import (
    RCNNBaseline,
    SyntheticDataGenerator,
    generate_weekly_precipitation,
    invRietkerk,
)


# The training script `train_invPDE_synthetic_batch.py` runs its batch
# training loop at module scope (no `if __name__ == "__main__"` guard), so
# importing from it would kick off a full training run. Instead we subclass
# the non-trainable `invRietkerk` from `train_rcnn_batch` and promote the 9
# scalar ecological coefficients to `nn.Parameter`s, which is exactly the
# form the saved state dicts use.
class _InvPDELearned(invRietkerk):
    """invRietkerk with scalar ecological coefficients registered as
    nn.Parameters of shape (1,), matching the state_dict format saved by
    train_invPDE_synthetic_batch.py (`model.state_dict()` with trainable=True)."""

    _SCALAR_NAMES = (
        "infiltration_rate",
        "seepage_rate",
        "plant_uptake_rate",
        "mortality_rate",
        "evaporation_rate",
        "water_use_efficiency",
        "surface_water_diffusion_coeff",
        "soil_water_diffusion_coeff",
        "biomass_diffusion_coeff",
    )

    def __init__(self):
        super().__init__(trainable=False)
        for name in self._SCALAR_NAMES:
            val = float(getattr(self, name))
            # Drop the plain-float attribute set by the parent, then register
            # a proper nn.Parameter so load_state_dict will populate it.
            object.__delattr__(self, name)
            self.register_parameter(name, nn.Parameter(torch.tensor([val])))
#%%
# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
RETRAIN_DIR = os.path.join("results", "synthetic_rcnn_4site_retrain")
INVPDE_DIR = os.path.join("results", "synthetic_invPDE_4site")
OUT_DIR = os.path.join(RETRAIN_DIR, "extrapolation_test")
os.makedirs(OUT_DIR, exist_ok=True)

GRID_SIZE = (128, 128)              # matches train_{rcnn,invPDE_synthetic}_batch
DATA_SEED = 42                      # must match training
EQUILIBRIUM_PRECIP = 19.0           # must match training
EQUILIBRIUM_YEARS = 100
NUM_YEARS = 100                     # extrapolation horizon (training used 10)
PDE_STEPS_PER_WEEK = 1              # matches generate_training_data in both scripts

# Precipitation regimes to test (annual mm). Training covered 16.5..24.5
# (4 sites linearly spaced). Test one below, one in the middle, one above.
REGIMES = [
    dict(label="below range", annual=14.5, peak_week=26.0, amp=0.7),
    dict(label="in range",    annual=20.5, peak_week=26.0, amp=0.7),
    dict(label="above range", annual=25.5, peak_week=26.0, amp=0.7),
]
#%%

# ---------------------------------------------------------------------------
# Best-model selection
# ---------------------------------------------------------------------------
def find_best_run():
    """Return (run_id, seed, final_loss) of the best run across all summary JSONs."""
    summary_files = sorted(glob.glob(os.path.join(RETRAIN_DIR, "summary_*.json")))
    if not summary_files:
        raise FileNotFoundError(f"No summary_*.json in {RETRAIN_DIR}")
    all_runs = []
    for sf in summary_files:
        with open(sf) as f:
            all_runs.extend(json.load(f))
    best = min(all_runs, key=lambda r: r["final_loss"])
    print(f"Checked {len(all_runs)} runs across {len(summary_files)} summary files.")
    print(f"Best run: {best['run_id']:02d} (seed={best['seed']}, "
          f"final_loss={best['final_loss']:.4f})")
    return best


def load_rcnn(run_id, device):
    ckpt_path = os.path.join(RETRAIN_DIR, f"run_{run_id:02d}.pt")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = RCNNBaseline(hidden_channels=32, num_layers=1).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def find_best_invpde_run():
    """Return the best inverse-PDE run (min finite final_loss).

    Runs whose final_loss is NaN/inf (training diverged) are skipped — the
    training script appends the NaN loss before breaking out of the loop,
    so those entries show up in summaries but are unusable as models.
    """
    summary_files = sorted(glob.glob(os.path.join(INVPDE_DIR, "summary_*.json")))
    if not summary_files:
        raise FileNotFoundError(f"No summary_*.json in {INVPDE_DIR}")
    all_runs = []
    for sf in summary_files:
        with open(sf) as f:
            all_runs.extend(json.load(f))
    # Some summary files overlap (e.g. 00-01 and 00-09); deduplicate on run_id,
    # keeping the lowest finite final_loss per run.
    best_per_run = {}
    for r in all_runs:
        rid = r["run_id"]
        loss = r.get("final_loss", float("nan"))
        if not np.isfinite(loss):
            continue
        if rid not in best_per_run or loss < best_per_run[rid]["final_loss"]:
            best_per_run[rid] = r
    if not best_per_run:
        raise RuntimeError("No invPDE runs with finite final_loss in summaries.")
    diverged = sorted({r["run_id"] for r in all_runs
                       if not np.isfinite(r.get("final_loss", float("nan")))})
    best = min(best_per_run.values(), key=lambda r: r["final_loss"])
    print(f"InvPDE: {len(best_per_run)} usable runs "
          f"across {len(summary_files)} summary files "
          f"({len(diverged)} NaN/inf skipped: {diverged}).")
    print(f"Best invPDE run: {best['run_id']:02d} (seed={best['seed']}, "
          f"final_loss={best['final_loss']:.4f})")
    return best


def load_invpde(run_id, device):
    ckpt_path = os.path.join(INVPDE_DIR, "models", f"invPDE_run_{run_id:02d}.pt")
    model = _InvPDELearned().to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    # Training script saves model.state_dict() directly.
    model.load_state_dict(state)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Simulation helpers
# ---------------------------------------------------------------------------
def make_initial_state(device):
    torch.manual_seed(DATA_SEED)
    np.random.seed(DATA_SEED)
    gen = SyntheticDataGenerator(grid_size=GRID_SIZE, device=device)
    biomass = gen.generate_equilibrium_state(
        equilibrium_precipitation=EQUILIBRIUM_PRECIP,
        equilibrium_years=EQUILIBRIUM_YEARS,
        steps_per_week=3,
    )
    # PDE needs surface/soil water too; start them at zeros then let it settle
    # through the first simulated year.
    sw = torch.zeros_like(biomass)
    soil = torch.zeros_like(biomass)
    return sw, soil, biomass


@torch.no_grad()
def rollout_rcnn(model, biomass0, weekly_precip, num_years, device):
    """Returns array of shape (num_years+1,) with mean biomass per year end."""
    biomass = biomass0.clone().to(device)
    precip_map = torch.full(
        (biomass.shape[0], 1, *GRID_SIZE),
        float(np.mean(weekly_precip)),  # spatial uniform; value reset weekly below
        device=device,
    )
    means = [biomass.mean().item()]
    hidden = None
    precip_tensor = torch.tensor(weekly_precip, device=device, dtype=biomass.dtype)
    for _ in range(num_years):
        biomass, hidden = model.simulate_year_weekly(biomass, precip_tensor, hidden)
        means.append(biomass.mean().item())
    final_field = biomass.squeeze().cpu().numpy()
    return np.array(means), final_field


@torch.no_grad()
def rollout_pde(pde, sw0, soil0, biomass0, weekly_precip, num_years, device):
    sw, soil, biomass = sw0.clone(), soil0.clone(), biomass0.clone()
    means = [biomass.mean().item()]
    precip_tensor = torch.tensor(weekly_precip, device=device, dtype=biomass.dtype)
    for _ in range(num_years):
        sw, soil, biomass = pde.simulate_year_weekly(
            sw, soil, biomass, precip_tensor, steps_per_week=PDE_STEPS_PER_WEEK
        )
        means.append(biomass.mean().item())
    final_field = biomass.squeeze().cpu().numpy()
    return np.array(means), final_field

#%%
# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

best = find_best_run()
rcnn = load_rcnn(best["run_id"], device)
pde = invRietkerk(trainable=False).to(device)
pde.eval()

best_invpde = find_best_invpde_run()
invpde = load_invpde(best_invpde["run_id"], device)

sw0, soil0, biomass0 = make_initial_state(device)
print(f"Initial biomass mean: {biomass0.mean().item():.4f}")

results = []
for reg in REGIMES:
    print(f"\nRegime: {reg['label']}  "
            f"(annual={reg['annual']}, peak_week={reg['peak_week']}, amp={reg['amp']})")
    weekly = generate_weekly_precipitation(
        annual_total=reg["annual"],
        peak_week=reg["peak_week"],
        amplitude_fraction=reg["amp"],
    )
    rcnn_traj, rcnn_field = rollout_rcnn(rcnn, biomass0, weekly, NUM_YEARS, device)
    pde_traj, pde_field = rollout_pde(pde, sw0, soil0, biomass0, weekly, NUM_YEARS, device)
    invpde_traj, invpde_field = rollout_pde(
        invpde, sw0, soil0, biomass0, weekly, NUM_YEARS, device
    )
    err_rcnn = np.abs(rcnn_traj - pde_traj)
    err_invpde = np.abs(invpde_traj - pde_traj)
    print(f"  PDE (truth) final mean biomass: {pde_traj[-1]:.4f}")
    print(f"  InvPDE     final mean biomass: {invpde_traj[-1]:.4f}  "
            f"(|err|={err_invpde[-1]:.4f}, max={err_invpde.max():.4f})")
    print(f"  RCNN       final mean biomass: {rcnn_traj[-1]:.4f}  "
            f"(|err|={err_rcnn[-1]:.4f}, max={err_rcnn.max():.4f})")
    results.append(dict(regime=reg, rcnn=rcnn_traj, pde=pde_traj,
                        invpde=invpde_traj,
                        rcnn_field=rcnn_field, pde_field=pde_field,
                        invpde_field=invpde_field))

# ---------------- Plot ----------------
#%%
n = len(results)
ncols = 3
nrows = int(np.ceil(n / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows),
                            sharex=True)
axes = np.array(axes).flatten()
years = np.arange(NUM_YEARS + 1)
for i, r in enumerate(results):
    ax = axes[i]
    ax.plot(years, r["pde"], "k-", label="PDE ground truth", linewidth=2)
    ax.plot(years, r["invpde"], "C0--", label="PDE (learned)", linewidth=2)
    ax.plot(years, r["rcnn"], "C1--", label="RCNN", linewidth=2)
    reg = r["regime"]
    ax.set_title(f"{reg['label']}\n"
                    f"annual={reg['annual']}, peak={reg['peak_week']:.0f}, "
                    f"amp={reg['amp']}", fontsize=10)
    ax.set_xlabel("year")
    ax.set_ylabel("mean biomass")
    ax.grid(alpha=0.3)
    ax.axvline(10, color="gray", linestyle=":", linewidth=1,
                label="train horizon")
    ax.legend(fontsize=7)
for j in range(n, len(axes)):
    axes[j].axis("off")
fig.suptitle(
    f"Extrapolation: RCNN run {best['run_id']:02d} "
    f"(loss {best['final_loss']:.3f}), "
    f"InvPDE run {best_invpde['run_id']:02d} "
    f"(loss {best_invpde['final_loss']:.3f}) vs PDE ground truth",
    fontsize=12,
)
fig.tight_layout()
out = os.path.join(OUT_DIR, "rcnn_vs_pde_extrapolation.png")
fig.savefig(out, dpi=150)
print(f"\nSaved {out}")

# ---------------- Final-year spatial snapshots ----------------
# Layout matches the reference figure: rows = {Ground Truth, PDE Model,
# RCNN Model}, cols = precipitation regimes. One shared colorbar per row.
nc = len(results)
row_specs = [
    ("Ground truth", "pde_field"),
    ("Inverse PDE",    "invpde_field"),
    ("RCNN",   "rcnn_field"),
]
# One extra narrow column for each row's colorbar.
fig2, ax2 = plt.subplots(
    len(row_specs), nc + 1,
    figsize=(3.3 * nc + 0.8, 3.3 * len(row_specs)),
    gridspec_kw={"width_ratios": [1] * nc + [0.06]},
    squeeze=False,
)
# Common colour scale across all panels so mean/MSE are visually comparable.
vmax = max(r["pde_field"].max() for r in results)
vmax = 15     # clamp to the reference's 0..2 scale
vmin = 0.0
precip_regimes = ["Low precipitation (spots)", "Mid precipitation (labyrinth)", "High precipitation (gaps)"]
for row_idx, (row_label, field_key) in enumerate(row_specs):
    for col_idx, r in enumerate(results):
        ax = ax2[row_idx, col_idx]
        field = r[field_key]
        im = ax.imshow(field, vmin=vmin, vmax=vmax, cmap="YlGn")
        ax.set_xticks([]); ax.set_yticks([])

        mean_v = float(field.mean())
        if row_idx == 0:
            ax.text(
                0.5, 1.15, precip_regimes[col_idx],
                transform=ax.transAxes, fontsize=10,
                fontweight="bold", ha="center", va="bottom"
            )
            ax.text(
                0.5, 1.02, f"Mean biomass: {mean_v:.3f}",
                transform=ax.transAxes, fontsize=10,
                ha="center", va="bottom"
            )
        else:
            # Model rows: Mean + MSE vs ground truth, above the panel.
            mse = float(((field - r["pde_field"]) ** 2).mean())
            ax.set_title(f"Mean biomass: {mean_v:.3f}\nMSE: {mse:.4f}", fontsize=10)

    # Left-edge row label.
    ax2[row_idx, 0].set_ylabel(row_label, fontsize=12, rotation=90,
                                labelpad=10)
    # Shared colorbar for the row in the extra column.
    fig2.colorbar(im, cax=ax2[row_idx, -1])

fig2.tight_layout()
out2 = os.path.join(OUT_DIR, "final_year_snapshots.png")
fig2.savefig(out2, dpi=150)
print(f"Saved {out2}")

# Also save raw field arrays
np.savez(
    os.path.join(OUT_DIR, "final_year_fields.npz"),
    **{f"pde_{r['regime']['label'].replace(' ', '_')}": r["pde_field"]
        for r in results},
    **{f"invpde_{r['regime']['label'].replace(' ', '_')}": r["invpde_field"]
        for r in results},
    **{f"rcnn_{r['regime']['label'].replace(' ', '_')}": r["rcnn_field"]
        for r in results},
)
plt.show()


# %%
