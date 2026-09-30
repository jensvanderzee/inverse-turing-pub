

#%%
import os
import re
from glob import glob

import contextily as cx
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from pyproj import Transformer
#%%
DATA_DIR = "data"
OUT_PATH = os.path.join("results", "real_data", "site_map.png")

# Site categories (from realdata_train_invPDE.py / realdata_test_invPDE.py).
TRAINING = {"b", "i", "c", "e"}
TEST     = {"f", "j", "k"}
VALIDATION = {"a"}

STYLE = {
    "training":   {"edgecolor": "#1f4fff", "label": "Training sites"},
    "test":       {"edgecolor": "#e040a0", "label": "Test sites"},
    "validation": {"edgecolor": "#2ecc2e", "label": "Validation site"},
}

NIAMEY_LONLAT = (2.1254, 13.5116)

# Transformers: WGS84 <-> Web Mercator, UTM31N -> WGS84.
to_merc    = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
utm31_to_ll = Transformer.from_crs("EPSG:32631", "EPSG:4326", always_xy=True)


def parse_aoi(path):
    """Return (xs, ys) arrays from an aoi file of comma-separated [x,y] pairs."""
    with open(path) as f:
        txt = f.read()
    pairs = re.findall(r"\[\s*([\-0-9.eE]+)\s*,\s*([\-0-9.eE]+)\s*\]", txt)
    xs = np.array([float(x) for x, _ in pairs])
    ys = np.array([float(y) for _, y in pairs])
    return xs, ys


def load_site_bboxes(data_dir=DATA_DIR):
    """Return {site_letter: (lon_min, lat_min, lon_max, lat_max)} in WGS84."""
    bboxes = {}
    for site_dir in sorted(glob(os.path.join(data_dir, "subsite_*"))):
        letter = os.path.basename(site_dir).split("_")[-1]
        aoi = None
        for name in ("aoi", "aoi.txt"):
            cand = os.path.join(site_dir, name)
            if os.path.isfile(cand):
                aoi = cand
                break
        if aoi is None:
            print(f"  skip {letter}: no aoi file")
            continue

        xs, ys = parse_aoi(aoi)
        # Heuristic: anything far outside a lat/lon range is projected.
        if np.max(np.abs(xs)) > 360 or np.max(np.abs(ys)) > 90:
            lon, lat = utm31_to_ll.transform(xs, ys)
        else:
            lon, lat = xs, ys
        bboxes[letter] = (lon.min(), lat.min(), lon.max(), lat.max())
    return bboxes


def categorize(letter):
    if letter in TRAINING:   return "training"
    if letter in TEST:       return "test"
    if letter in VALIDATION: return "validation"
    return None


def _add_north_arrow(ax, x=0.07, y=0.93, size=0.06):
    """Simple N arrow in axes fraction coords."""
    ax.annotate(
        "",
        xy=(x, y), xycoords="axes fraction",
        xytext=(x, y - size),
        arrowprops=dict(arrowstyle="-|>", color="black", lw=2.2, mutation_scale=22),
    )
    ax.text(x, y + 0.012, "N", transform=ax.transAxes,
            ha="center", va="bottom", fontsize=14, fontweight="bold")


def _add_scalebar(ax, length_km=20, lat_deg=14.0, loc=(0.62, 0.06)):
    """Scale bar in Web Mercator coords, corrected for latitude."""
    # 1 meter in Web Mercator at latitude lat_deg:
    m_per_merc = np.cos(np.deg2rad(lat_deg))
    length_m = length_km * 1000 / m_per_merc

    x0, y0 = ax.transAxes.transform(loc)
    x0, y0 = ax.transData.inverted().transform((x0, y0))

    # Draw three segments (0, 10, 20 km) with alternating black/white fill.
    seg = length_m / 2
    bar_h = seg * 0.08
    for i, col in enumerate(["black", "white"]):
        ax.add_patch(mpatches.Rectangle(
            (x0 + i * seg, y0), seg, bar_h,
            facecolor=col, edgecolor="black", lw=1.0, zorder=6,
        ))
    for i, label in enumerate(("0", f"{length_km // 2}", f"{length_km}")):
        ax.text(x0 + i * seg, y0 - bar_h * 1.6, label,
                ha="center", va="top", fontsize=9, zorder=6)
    ax.text(x0 + length_m + bar_h, y0 + bar_h * 0.3, "km",
            ha="left", va="center", fontsize=9, zorder=6)


#%%
bboxes = load_site_bboxes()
print(f"Loaded {len(bboxes)} site AOIs: {sorted(bboxes)}")

# Overall lon/lat extent with padding so Niamey and all sites fit.
lons = [v for b in bboxes.values() for v in (b[0], b[2])] + [NIAMEY_LONLAT[0]]
lats = [v for b in bboxes.values() for v in (b[1], b[3])] + [NIAMEY_LONLAT[1]]
pad_lon = 0.05
pad_lat = 0.05
lon_min, lon_max = min(lons) - pad_lon, max(lons) + pad_lon
lat_min, lat_max = min(lats) - pad_lat, max(lats) + pad_lat

# Reproject map extent to Web Mercator for contextily.
x0, y0 = to_merc.transform(lon_min, lat_min)
x1, y1 = to_merc.transform(lon_max, lat_max)
#%%
fig, ax = plt.subplots(figsize=(7.5, 10))
ax.set_xlim(x0, x1)
ax.set_ylim(y0, y1)
ax.set_aspect("equal")
# Degree ticks, labeled in lon/lat but positioned in Web Mercator.
lon_ticks = np.arange(np.ceil(lon_min * 2) / 2, lon_max, 0.25)  # every 0.5°
lat_ticks = np.arange(np.ceil(lat_min * 2) / 2, lat_max, 0.25)
x_ticks = [to_merc.transform(lon, lat_min)[0] for lon in lon_ticks]
y_ticks = [to_merc.transform(lon_min, lat)[1] for lat in lat_ticks]
ax.set_xticks(x_ticks)
ax.set_yticks(y_ticks)
ax.set_xticklabels([f"{lon:.1f}°E" for lon in lon_ticks], fontsize=9)
ax.set_yticklabels([f"{lat:.1f}°N" for lat in lat_ticks], fontsize=9)
ax.tick_params(direction="in", length=4, colors="black")

# Basemap: Esri World Imagery (satellite).
# cx.add_basemap(ax, source=cx.providers.Esri.WorldImagery, crs="EPSG:3857",
#                 attribution_size=6)
cx.add_basemap(ax, 
               source=cx.providers.Esri.WorldImagery, 
               crs="EPSG:3857",
               zoom=11,  # Manually set this
               attribution_size=6)
# Site rectangles.
for letter, (lo_min, la_min, lo_max, la_max) in bboxes.items():
    cat = categorize(letter)
    if cat is None:
        continue
    mx0, my0 = to_merc.transform(lo_min, la_min)
    mx1, my1 = to_merc.transform(lo_max, la_max)
    ax.add_patch(mpatches.Rectangle(
        (mx0, my0), mx1 - mx0, my1 - my0,
        fill=False, edgecolor=STYLE[cat]["edgecolor"],
        linewidth=2.2, zorder=5,
    ))

# Niamey label.
nx, ny = to_merc.transform(*NIAMEY_LONLAT)
ax.text(nx, ny, "Niamey", color="black", fontsize=11, fontweight="bold",
        ha="left", va="center",
        path_effects=None, zorder=6)

_add_north_arrow(ax)
_add_scalebar(ax, length_km=20, lat_deg=(lat_min + lat_max) / 2, loc=(0.81, 0.06))

# Legend.
handles = [mpatches.Patch(facecolor="none", edgecolor=s["edgecolor"],
                            linewidth=2.2, label=s["label"])
            for s in STYLE.values()]
leg = ax.legend(handles=handles, loc="lower left",
                bbox_to_anchor=(0.0, 0.04),  # nudge up ~8% of axes height
                frameon=True, framealpha=0.95, fontsize=10)
leg.get_frame().set_edgecolor("black")

os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
fig.tight_layout()
fig.savefig(OUT_PATH, dpi=300, bbox_inches="tight")
print(f"Saved {OUT_PATH}")
plt.show()


# %%
