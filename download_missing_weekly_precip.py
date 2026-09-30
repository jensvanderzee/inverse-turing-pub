# -*- coding: utf-8 -*-
"""
Download missing weekly precipitation data for subsites that only have
annual totals (currently: subsite_j and subsite_k).

Step 1: Extract bounding-box coordinates from the existing GeoTIFF files
        and create AOI files so future scripts can reuse them.
Step 2: Fetch ERA5 daily precipitation via Google Earth Engine.
Step 3: Aggregate into weekly averages (mm/day) — same format as the
        existing weekly CSVs.

Usage:
    python download_missing_weekly_precip.py

Prerequisites:
    pip install earthengine-api pandas numpy rasterio
    Authenticated via:  earthengine authenticate
"""
#%%
import os
import glob

import numpy as np
import rasterio

# Reuse the GEE fetch + weekly aggregation from the existing script
from download_era5_precip_gee import fetch_and_process_weekly_csv

# ════════════════════════════════════════════════════════════════════════════
# SETTINGS
# ════════════════════════════════════════════════════════════════════════════
SUBSITES = ["subsite_j", "subsite_k"]   # subsites to process
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
YEARS    = list(range(2013, 2024))       # match existing data range
# The CSVs first written for j and k came from a collection that ends in July 2020,
# with the gaps filled as zero rain (all of k, 2020-2023 for j). Re-fetch them.
OVERWRITE = True


def extract_bbox_from_tiffs(ndvi_dir: str) -> dict:
    """
    Read the georeferencing metadata from the first GeoTIFF in a
    subsite's NDVI directory and return a bounding box in EPSG:4326
    (longitude / latitude degrees).

    Handles TIFFs stored in any CRS (e.g. UTM) by reprojecting the
    bounds to WGS-84.
    """
    from rasterio.warp import transform_bounds

    tif_files = sorted(
        glob.glob(os.path.join(ndvi_dir, "*.tif"))
        + glob.glob(os.path.join(ndvi_dir, "*.tiff"))
    )
    if not tif_files:
        raise FileNotFoundError(f"No TIFF files found in {ndvi_dir}")

    with rasterio.open(tif_files[0]) as src:
        src_crs = src.crs
        bounds = src.bounds          # BoundingBox(left, bottom, right, top)

    # Reproject bounds to WGS-84 if needed
    if src_crs and str(src_crs) != "EPSG:4326":
        print(f"  TIFF CRS is {src_crs}, reprojecting bounds to EPSG:4326")
        west, south, east, north = transform_bounds(
            src_crs, "EPSG:4326",
            bounds.left, bounds.bottom, bounds.right, bounds.top,
        )
    else:
        west, south, east, north = bounds.left, bounds.bottom, bounds.right, bounds.top

    return {
        "north": north,
        "south": south,
        "east":  east,
        "west":  west,
    }


def write_aoi_file(bbox: dict, aoi_path: str):
    """Write an AOI file in the same [lon,lat] polygon format used elsewhere."""
    sw = f"[{bbox['west']},{bbox['south']}]"
    nw = f"[{bbox['west']},{bbox['north']}]"
    ne = f"[{bbox['east']},{bbox['north']}]"
    se = f"[{bbox['east']},{bbox['south']}]"
    polygon = f"{sw},{nw},{ne},{se},{sw}"

    with open(aoi_path, "w") as f:
        f.write(polygon)
    print(f"  AOI file written: {aoi_path}")


def main():
    for subsite_name in SUBSITES:
        subsite_dir = os.path.join(DATA_DIR, subsite_name)
        if not os.path.isdir(subsite_dir):
            print(f"Skipping {subsite_name}: directory not found")
            continue

        print(f"\n{'=' * 60}")
        print(f"Processing {subsite_name}")
        print(f"{'=' * 60}")

        # ── Step 1: extract bounding box and create AOI file ────────────
        ndvi_dir = os.path.join(subsite_dir, f"{subsite_name}_ndvi")
        bbox = extract_bbox_from_tiffs(ndvi_dir)
        print(f"  Bounding box from TIFF: {bbox}")

        aoi_path = os.path.join(subsite_dir, "aoi.txt")
        if not os.path.exists(aoi_path):
            write_aoi_file(bbox, aoi_path)
        else:
            print(f"  AOI file already exists: {aoi_path}")

        # ── Step 2+3: fetch from GEE and write weekly CSV ──────────────
        precip_dir = os.path.join(subsite_dir, f"{subsite_name}_precip")
        os.makedirs(precip_dir, exist_ok=True)
        csv_path = os.path.join(precip_dir, f"{subsite_name}_weekly_precip.csv")

        if os.path.exists(csv_path) and not OVERWRITE:
            print(f"  Weekly CSV already exists, skipping: {csv_path}")
        else:
            fetch_and_process_weekly_csv(subsite_name, bbox, csv_path, YEARS)

    print(f"\n{'=' * 60}")
    print("Done! Weekly precipitation CSVs created.")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()

# %%
