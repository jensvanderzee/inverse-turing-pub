# -*- coding: utf-8 -*-
"""
Download complete weekly precipitation for the held-out subsites j and k.

The weekly CSVs first written for these sites were broken:
  - subsite_k: its box straddles a grid-cell edge, Earth Engine returned no value
    for any day, and the gaps were written as zero rain (every year).
  - subsite_j: taken from ECMWF/ERA5/DAILY, which ends on 2020-07-09, so 2020 on
    was written as zero rain; 2013-2019 also came from that collection rather
    than the one the training sites use.

This script rebuilds both from the same source and processing as the training
sites' weekly CSVs (verified by reproducing subsite_b to ~1e-16 mm/day):
  - daily totals = sum of ECMWF/ERA5/HOURLY `total_precipitation` over each UTC day
  - ERA5 sampled on the EPSG:4326 grid at scale 27830 m (cell edges on multiples of
    ~0.25 deg), i.e. what reduceRegion(mean, scale=27830) uses; a box covering
    several cells gets their area-weighted mean
  - week w = days 7(w-1)+1 .. 7w of the year, days 365/366 folded into week 52;
    each week is the mean daily rate in mm/day

A missing or null day raises an error instead of being written as zero rain.

Usage (needs only `earthengine-api`; runs in the `dataio` environment):
    python download_missing_weekly_precip.py --project <earth-engine-cloud-project>
    python download_missing_weekly_precip.py --project <id> --out-dir <dir>   # don't touch data/

The project can also be given with the EARTHENGINE_PROJECT environment variable.
Authenticate once with `earthengine authenticate`.
"""
#%%
import argparse
import csv
import datetime as dt
import math
import os
import re

import ee

# ════════════════════════════════════════════════════════════════════════════
# SETTINGS
# ════════════════════════════════════════════════════════════════════════════
SUBSITES = ["subsite_j", "subsite_k"]   # subsites to process
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
YEARS    = list(range(2013, 2024))       # match existing data range

COLLECTION = "ECMWF/ERA5/HOURLY"
BAND       = "total_precipitation"       # metres per hour
GRID_SCALE_M = 27830                     # grid of the training sites' weekly CSVs


def read_bbox(subsite_dir: str) -> tuple:
    """(west, south, east, north) in degrees from the subsite's [lon,lat] aoi file."""
    for name in ("aoi", "aoi.txt"):
        path = os.path.join(subsite_dir, name)
        if os.path.exists(path):
            break
    else:
        raise FileNotFoundError(f"No aoi file in {subsite_dir}")
    with open(path) as f:
        pairs = re.findall(r"\[([-\d.eE]+),([-\d.eE]+)\]", f.read())
    lons = [float(p[0]) for p in pairs]
    lats = [float(p[1]) for p in pairs]
    if max(map(abs, lons)) > 180 or max(map(abs, lats)) > 90:
        raise ValueError(f"{path} holds projected coordinates, not lon/lat degrees")
    return min(lons), min(lats), max(lons), max(lats)


def cell_weights(bbox: tuple, cell: float) -> list:
    """[(lon, lat, weight)] for each grid cell the box overlaps: the cell centre and
    the fraction of the box's area inside it. The weights sum to exactly 1."""
    west, south, east, north = bbox

    def overlaps(lo, hi):
        return [(i, min(hi, (i + 1) * cell) - max(lo, i * cell))
                for i in range(math.floor(lo / cell), math.floor(hi / cell) + 1)
                if min(hi, (i + 1) * cell) > max(lo, i * cell)]

    area = (east - west) * (north - south)
    cells = [((ix + 0.5) * cell, (iy + 0.5) * cell, wx * wy / area)
             for ix, wx in overlaps(west, east) for iy, wy in overlaps(south, north)]
    lon, lat, _ = cells[-1]
    cells[-1] = (lon, lat, 1.0 - sum(w for _, _, w in cells[:-1]))
    return cells


def fetch_daily_mm(bbox: tuple, year: int) -> list:
    """[(date, mm)] for every day of `year`: ERA5 daily total over the box."""
    grid = ee.Projection("EPSG:4326").atScale(GRID_SCALE_M)
    cells = cell_weights(bbox, grid.getInfo()["transform"][0])
    points = [ee.Geometry.Point([lon, lat]) for lon, lat, _ in cells]
    hourly = ee.ImageCollection(COLLECTION).select(BAND)
    start = ee.Date.fromYMD(year, 1, 1)
    n_days = (dt.date(year + 1, 1, 1) - dt.date(year, 1, 1)).days

    def one_day(i):
        day = start.advance(i, "day")
        img = hourly.filterDate(day, day.advance(1, "day")).sum().reproject(grid)
        values = [img.reduceRegion(ee.Reducer.first(), p, crs=grid).get(BAND) for p in points]
        return ee.Feature(None, {"date": day.format("YYYY-MM-dd"),
                                 "n_hours": hourly.filterDate(day, day.advance(1, "day")).size(),
                                 "values": values})

    feats = ee.FeatureCollection(ee.List.sequence(0, n_days - 1).map(one_day)).getInfo()["features"]
    out = []
    for f in feats:
        p = f["properties"]
        if p["n_hours"] != 24 or any(v is None for v in p["values"]):
            raise ValueError(f"{COLLECTION} is incomplete on {p['date']}: "
                             f"{p['n_hours']} hourly images, cell values {p['values']}")
        m = sum(w * v for (_, _, w), v in zip(cells, p["values"]))
        out.append((dt.date.fromisoformat(p["date"]), m * 1000.0))
    if [d for d, _ in out] != [dt.date(year, 1, 1) + dt.timedelta(i) for i in range(n_days)]:
        raise ValueError(f"{COLLECTION} returned {len(out)} of {n_days} days for {year}")
    return out


def weekly_rates(daily: list) -> list:
    """[(year, week, mean mm/day)], 52 weeks per year."""
    groups = {}
    for day, mm in daily:
        week = min((day.timetuple().tm_yday - 1) // 7 + 1, 52)
        groups.setdefault((day.year, week), []).append(mm)
    return [(y, w, sum(v) / len(v)) for (y, w), v in sorted(groups.items())]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--project", default=os.environ.get("EARTHENGINE_PROJECT"),
                        help="Earth Engine Cloud project ID")
    parser.add_argument("--out-dir", default=None,
                        help="write the CSVs here instead of into data/<subsite>/<subsite>_precip/")
    args = parser.parse_args()
    ee.Initialize(project=args.project)

    for subsite_name in SUBSITES:
        subsite_dir = os.path.join(DATA_DIR, subsite_name)
        bbox = read_bbox(subsite_dir)
        print(f"\n{subsite_name}: box W,S,E,N = {bbox}")

        daily = []
        for year in YEARS:
            daily += fetch_daily_mm(bbox, year)
            print(f"  {year}: {sum(mm for d, mm in daily if d.year == year):7.1f} mm", flush=True)
        weekly = weekly_rates(daily)
        assert len(weekly) == 52 * len(YEARS)

        out_dir = args.out_dir or os.path.join(subsite_dir, f"{subsite_name}_precip")
        os.makedirs(out_dir, exist_ok=True)
        csv_path = os.path.join(out_dir, f"{subsite_name}_weekly_precip.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f, lineterminator="\n")
            writer.writerow(["year", "week", "precipitation_mm_per_day"])
            writer.writerows((y, w, repr(r)) for y, w, r in weekly)
        print(f"  written: {csv_path}")


if __name__ == "__main__":
    main()

# %%
