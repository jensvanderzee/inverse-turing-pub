
#%%
import os
import pickle

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from tqdm import tqdm

from realdata_train_invPDE import invRietkerk, RealDataLoader
#%%
# ── Plot ────────────────────────────────────────────────────────────────────
best_idx = model_ids.index(best_model_id)

fig, ax = plt.subplots(figsize=(12, 6))

# Grey lines for all other models
for i, mid in enumerate(model_ids):
    if mid == best_model_id:
        continue
    ax.plot(precip_values, results[i], color="silver", lw=0.8, alpha=0.6)

# Dummy for legend
ax.plot([], [], color="silver", lw=0.8, alpha=0.6, label="Other models")

# Best model in bold black
ax.plot(precip_values, results[best_idx], color="black", lw=2.5, label="Best model")

# ── Inset spatial images ────────────────────────────────────────────────────
# Colourmap for insets (same green-yellow as reference figure)
cmap = plt.cm.YlGn
vmax_inset = NDVI_TO_BIOMASS_MULTIPLIER * 0.35  # adjust to get good contrast

for p_val, spatial in sorted(best_spatial.items()):
    mean_val = spatial.mean()

    # Plot a red marker + vertical line at this precipitation value
    ax.plot(p_val, mean_val, "o", color="red", ms=6, zorder=5)
    ax.axvline(p_val, color="red", lw=0.5, alpha=0.4, zorder=1)

    # Render the spatial image as an inset
    norm_inset = Normalize(vmin=0, vmax=vmax_inset)
    rgba = cmap(norm_inset(spatial))
    thumb = OffsetImage(rgba, zoom=0.55)
    thumb.image.axes = ax

    # Place inset above the data point with some vertical offset
    y_offset = 50
    ab = AnnotationBbox(
        thumb,
        (p_val, mean_val),
        xybox=(0, y_offset),
        xycoords="data",
        boxcoords="offset points",
        frameon=True,
        bboxprops=dict(edgecolor="grey", lw=0.8),
        arrowprops=dict(arrowstyle="-", color="red", lw=0.8),
    )
    ax.add_artist(ab)

# ── Colour bar for the insets ───────────────────────────────────────────────
sm = plt.cm.ScalarMappable(cmap=cmap, norm=Normalize(vmin=0, vmax=vmax_inset))
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, pad=0.02, fraction=0.03)
cbar.set_label("Vegetation density [g/m$^2$]", fontsize=11)

# ── Labels ──────────────────────────────────────────────────────────────────
ax.set_xlabel("Average annual precipitation [mm]", fontsize=12)
ax.set_ylabel("Average vegetation density [g/m$^2$]", fontsize=12)
ax.legend(loc="upper left", fontsize=11)
ax.set_xlim(PRECIP_MIN, PRECIP_MAX)
ax.set_ylim(bottom=0)

plt.tight_layout()
fig.savefig(os.path.join(SAVE_DIR, "bifurcation_diagram.png"), dpi=200,
            bbox_inches="tight")
fig.savefig(os.path.join(SAVE_DIR, "bifurcation_diagram.pdf"),
            bbox_inches="tight")
plt.show()
plt.close(fig)

print(f"\nPlots saved to {SAVE_DIR}/")
print("  bifurcation_diagram.png")
print("  bifurcation_diagram.pdf")
print("  bifurcation_data.csv")