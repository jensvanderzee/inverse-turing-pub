#%%
import os
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

from model_selection import best_model_by_test_mse

# ── Paths (same as plot_bifurcation.py) ──────────────────────────────────────
DATA_CSV         = "results/real_data/bifurcation/bifurcation_data.csv"
TEST_METRICS_CSV = "results/real_data/test_results/test_metrics.csv"
SAVE_DIR         = "results/real_data/bifurcation"
SNAPSHOTS_PKL    = os.path.join(SAVE_DIR, "spatial_snapshots.pkl")

INSET_PRECIP_VALUES      = [267, 285, 294, 303, 318]
NDVI_TO_BIOMASS_MULTIPLIER = 1500.0

#%%
# ── Load data ─────────────────────────────────────────────────────────────────
sweep_df      = pd.read_csv(DATA_CSV, index_col=0)
precip_values = sweep_df.columns.astype(float).values
model_ids     = list(sweep_df.index)
results       = sweep_df.values

test_df  = pd.read_csv(TEST_METRICS_CSV)
best_id, _ = best_model_by_test_mse(test_df)
best_idx = model_ids.index(best_id)

with open(SNAPSHOTS_PKL, "rb") as f:
    spatial_snapshots = pickle.load(f)

#%%
# ── Plot ──────────────────────────────────────────────────────────────────────
cmap       = plt.cm.YlGn
vmax_inset = NDVI_TO_BIOMASS_MULTIPLIER * 0.3
norm_inset = Normalize(vmin=0, vmax=vmax_inset)

fig, ax = plt.subplots(figsize=(12, 8))

for i, mid in enumerate(model_ids):
    if mid == best_id:
        continue
    ax.plot(precip_values, results[i], color="silver", lw=0.8, alpha=0.6)

ax.plot(precip_values, results[best_idx], color="black", lw=2.5)

# ── Inset spatial snapshots ───────────────────────────────────────────────────
inset_precip_sorted = sorted(p for p in spatial_snapshots if p in INSET_PRECIP_VALUES)

for rank, p_val in enumerate(inset_precip_sorted):
    spatial = spatial_snapshots[p_val]

    col_idx  = np.argmin(np.abs(precip_values - float(p_val)))
    mean_val = results[best_idx, col_idx]

    ax.plot(precip_values[col_idx], mean_val, "o", color="red", ms=6, zorder=5)

    rgba  = cmap(norm_inset(spatial))
    thumb = OffsetImage(rgba, zoom=0.55)
    thumb.image.axes = ax

    is_last  = (rank == len(inset_precip_sorted) - 1)
    y_offset = 81 if is_last else 145

    ab = AnnotationBbox(
        thumb,
        (precip_values[col_idx], mean_val),
        xybox=(0, y_offset),
        xycoords="data",
        boxcoords="offset points",
        frameon=True,
        bboxprops=dict(edgecolor="black", lw=0.8),
        arrowprops=dict(arrowstyle="-", color="red", lw=0.8),
    )
    ax.add_artist(ab)

# ── Strip all decorations ─────────────────────────────────────────────────────
ax.set_xlabel("")
ax.set_ylabel("")
ax.set_xlim(precip_values[0], 335)
ax.set_ylim(bottom=0)
ax.set_xticks([])
ax.set_yticks([])
#ax.spines[["top", "right", "left", "bottom"]].set_visible(False)

plt.tight_layout()
os.makedirs(SAVE_DIR, exist_ok=True)
fig.savefig(os.path.join(SAVE_DIR, "bifurcation_diagram_cosmetic.png"), dpi=200, bbox_inches="tight")
fig.savefig(os.path.join(SAVE_DIR, "bifurcation_diagram_cosmetic.pdf"), bbox_inches="tight")

plt.show()
plt.close(fig)

print(f"Saved to {SAVE_DIR}/bifurcation_diagram_cosmetic.png/.pdf")
# %%
