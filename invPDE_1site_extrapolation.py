
#%%
import glob
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from train_rcnn_batch import (
    SyntheticDataGenerator,
    generate_weekly_precipitation,
    invRietkerk,
)


class _InvPDELearned(invRietkerk):
    """invRietkerk with scalar ecological coefficients registered as
    nn.Parameters of shape (1,), matching the state_dict saved by
    train_invPDE_synthetic_batch.py (trainable=True)."""

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
            object.__delattr__(self, name)
            self.register_parameter(name, nn.Parameter(torch.tensor([val])))

#%%
# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
INVPDE_DIR = os.path.join("results", "synthetic_invPDE_1site")
OUT_DIR = os.path.join(INVPDE_DIR, "extrapolation_test")
os.makedirs(OUT_DIR, exist_ok=True)

GRID_SIZE = (128, 128)
DATA_SEED = 42
EQUILIBRIUM_PRECIP = 19.0
EQUILIBRIUM_YEARS = 100
NUM_YEARS = 1000
PDE_STEPS_PER_WEEK = 1

# Precipitation regimes to probe. 1-site training used a single annual
# precipitation value, so we sweep across a range to see how the learned
# model behaves outside that point.
REGIMES = [
    dict(label="below range", annual=14.5, peak_week=26.0, amp=0.7),
    dict(label="in range",    annual=20.5, peak_week=26.0, amp=0.7),
    dict(label="above range", annual=25.5, peak_week=26.0, amp=0.7),
]

#%%
# ---------------------------------------------------------------------------
# Best-model selection
# ---------------------------------------------------------------------------
def find_best_invpde_run():
    """Return the best 1-site inverse-PDE run (min finite final_loss)."""
    summary_files = sorted(glob.glob(os.path.join(INVPDE_DIR, "summary_*.json")))
    if not summary_files:
        raise FileNotFoundError(f"No summary_*.json in {INVPDE_DIR}")
    all_runs = []
    for sf in summary_files:
        with open(sf) as f:
            all_runs.extend(json.load(f))

    best_per_run = {}
    for r in all_runs:
        rid = r["run_id"]
        loss = r.get("final_loss", float("nan"))
        if not np.isfinite(loss):
            continue
        if rid not in best_per_run or loss < best_per_run[rid]["final_loss"]:
            best_per_run[rid] = r
    if not best_per_run:
        raise RuntimeError("No 1-site invPDE runs with finite final_loss.")

    diverged = sorted({r["run_id"] for r in all_runs
                       if not np.isfinite(r.get("final_loss", float("nan")))})
    best = min(best_per_run.values(), key=lambda r: r["final_loss"])
    print(f"1-site invPDE: {len(best_per_run)} usable runs "
          f"across {len(summary_files)} summary files "
          f"({len(diverged)} NaN/inf skipped: {diverged}).")
    print(f"Best run: {best['run_id']:02d} (seed={best['seed']}, "
          f"final_loss={best['final_loss']:.4f})")
    return best


def load_invpde(run_id, device):
    ckpt_path = os.path.join(INVPDE_DIR, "models", f"invPDE_run_{run_id:02d}.pt")
    model = _InvPDELearned().to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
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
    sw = torch.zeros_like(biomass)
    soil = torch.zeros_like(biomass)
    return sw, soil, biomass


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

best = find_best_invpde_run()
invpde = load_invpde(best["run_id"], device)

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
    traj, field = rollout_pde(invpde, sw0, soil0, biomass0, weekly, NUM_YEARS, device)
    print(f"  Final mean biomass: {traj[-1]:.4f}  (max across rollout: {traj.max():.4f})")
    results.append(dict(regime=reg, traj=traj, field=field))

#%%
# ---------------- Trajectory plot ----------------
n = len(results)
ncols = 3
nrows = int(np.ceil(n / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows),
                         sharex=True)
axes = np.array(axes).flatten()
years = np.arange(NUM_YEARS + 1)
for i, r in enumerate(results):
    ax = axes[i]
    ax.plot(years, r["traj"], "C0-", linewidth=2, label="1-site invPDE (learned)")
    reg = r["regime"]
    ax.set_title(f"{reg['label']}\n"
                 f"annual={reg['annual']}, peak={reg['peak_week']:.0f}, "
                 f"amp={reg['amp']}", fontsize=10)
    ax.set_xlabel("year")
    ax.set_ylabel("mean biomass")
    ax.grid(alpha=0.3)
    ax.axvline(10, color="gray", linestyle=":", linewidth=1,
               label="train horizon")
    ax.legend(fontsize=8)
for j in range(n, len(axes)):
    axes[j].axis("off")
fig.suptitle(
    f"Extrapolation: 1-site invPDE run {best['run_id']:02d} "
    f"(final loss {best['final_loss']:.3f})",
    fontsize=12,
)
fig.tight_layout()
out = os.path.join(OUT_DIR, "invPDE_1site_extrapolation.png")
fig.savefig(out, dpi=150)
print(f"\nSaved {out}")

#%%
# ---------------- Final-year spatial snapshots ----------------
nc = len(results)
fig2, ax2 = plt.subplots(
    1, nc + 1,
    figsize=(3.3 * nc + 0.8, 3.6),
    gridspec_kw={"width_ratios": [1] * nc + [0.06]},
    squeeze=False,
)
vmax = 14.0  # same clamp as 4-site extrapolation script for visual parity
vmin = 0.0
precip_titles = [f"{r['regime']['label']}\nannual={r['regime']['annual']} mm"
                 for r in results]
for col_idx, r in enumerate(results):
    ax = ax2[0, col_idx]
    im = ax.imshow(r["field"], vmin=vmin, vmax=vmax, cmap="YlGn")
    ax.set_xticks([]); ax.set_yticks([])
    mean_v = float(r["field"].mean())
    ax.set_title(f"{precip_titles[col_idx]}\nMean biomass: {mean_v:.3f}",
                 fontsize=10)
ax2[0, 0].set_ylabel(
    f"1-site invPDE\nrun {best['run_id']:02d}", fontsize=11, rotation=90, labelpad=10
)
fig2.colorbar(im, cax=ax2[0, -1])
fig2.tight_layout()
out2 = os.path.join(OUT_DIR, "invPDE_1site_final_year_snapshots.png")
fig2.savefig(out2, dpi=150)
print(f"Saved {out2}")

# Save raw fields and trajectories
np.savez(
    os.path.join(OUT_DIR, "invPDE_1site_results.npz"),
    years=years,
    **{f"traj_{r['regime']['label'].replace(' ', '_')}": r["traj"]
       for r in results},
    **{f"field_{r['regime']['label'].replace(' ', '_')}": r["field"]
       for r in results},
)
plt.show()

# %%
