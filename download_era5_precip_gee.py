# -*- coding: utf-8 -*-
"""
Fetch ERA5 daily total precipitation for each subsite using Google Earth Engine 
and compute weekly averages.

Run in your conda environment:
    python get_era5_precip_gee.py
"""
#%%
import os
import re
import glob
import pandas as pd
import numpy as np
import ee

# Initialize Google Earth Engine
try:
    ee.Initialize()
except Exception:
    print("Google Earth Engine not authorized. Prompting authentication...")
    ee.Authenticate()
    ee.Initialize()

# ERA5-Land daily aggregates. The older 'ECMWF/ERA5/DAILY' collection stops on
# 2020-07-09, which silently truncated the subsite_j and subsite_k downloads.
# `total_precipitation_sum` is also the column name in the annual totals, which
# suggests they came from ERA5-Land too.
ERA5_COLLECTION = 'ECMWF/ERA5_LAND/DAILY_AGGR'
ERA5_BAND = 'total_precipitation_sum'   # metres per day
ERA5_SCALE_M = 11132                    # native resolution, 0.1 degree


def parse_aoi(aoi_path: str) -> dict:
    """
    Parse an AOI file containing a bounding box as coordinate pairs.
    Format: [lon1,lat1],[lon2,lat2],...

    Returns dict with keys: north, south, east, west
    """
    with open(aoi_path, 'r') as f:
        content = f.read().strip()

    # Extract all [lon,lat] pairs
    pairs = re.findall(r'\[([-\d.]+),([-\d.]+)\]', content)
    if not pairs:
        raise ValueError(f"Could not parse coordinates from {aoi_path}: {content}")

    lons = [float(p[0]) for p in pairs]
    lats = [float(p[1]) for p in pairs]
    if max(abs(v) for v in lons) > 180 or max(abs(v) for v in lats) > 90:
        raise ValueError(f"{aoi_path} holds projected coordinates, not lon/lat degrees; "
                         f"use download_missing_weekly_precip.py, which reads the bbox from the GeoTIFFs")

    return {
        'north': max(lats),
        'south': min(lats),
        'east': max(lons),
        'west': min(lons),
    }


def fetch_and_process_weekly_csv(subsite_name: str, bbox: dict, output_csv: str, years: list):
    """
    Fetch daily spatial means from GEE, convert to mm, and compute weekly averages.
    """
    print(f"  Fetching ERA5 from GEE for {subsite_name}...")
    
    # 1. Define GEE Geometry and Dates
    geom = ee.Geometry.Rectangle([bbox['west'], bbox['south'], bbox['east'], bbox['north']])
    start_date = ee.Date.fromYMD(years[0], 1, 1)
    end_date = ee.Date.fromYMD(years[-1] + 1, 1, 1) # Exclusive end date

    # 2. Query the daily collection
    collection = ee.ImageCollection(ERA5_COLLECTION) \
        .select(ERA5_BAND) \
        .filterDate(start_date, end_date)

    # 3. Spatially average each daily image over the bounding box
    def get_spatial_mean(img):
        mean_dict = img.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=geom,
            scale=ERA5_SCALE_M,
            maxPixels=1e9
        )
        return ee.Feature(None, {
            'date': img.date().format('YYYY-MM-dd'),
            'precip_m': mean_dict.get(ERA5_BAND)
        })

    # Execute the reduction and fetch the time series data to the local machine
    # Note: getInfo() is fine here because ~11 years * 365 days = ~4000 records 
    # (well under the 5000 element limit for getInfo).
    ts_data = collection.map(get_spatial_mean).getInfo()
    
    # 4. Load into pandas
    records = [f['properties'] for f in ts_data['features']]
    daily_df = pd.DataFrame(records)
    
    # Missing data must not be written as zero rain: the model reads it as drought.
    daily_df['precip_m'] = pd.to_numeric(daily_df['precip_m'])
    daily_df['date'] = pd.to_datetime(daily_df['date'])
    daily_df = daily_df.sort_values('date').reset_index(drop=True)

    n_null = int(daily_df['precip_m'].isna().sum())
    if n_null:
        raise ValueError(f"{subsite_name}: {n_null} days with no value over the bounding box "
                         f"{bbox} (first: {daily_df.loc[daily_df['precip_m'].isna(), 'date'].iloc[0].date()})")
    expected = pd.date_range(f"{years[0]}-01-01", f"{years[-1]}-12-31", freq='D')
    missing = expected.difference(daily_df['date'])
    if len(missing):
        raise ValueError(f"{subsite_name}: {ERA5_COLLECTION} has no data for {len(missing)} of "
                         f"{len(expected)} days ({missing[0].date()} to {missing[-1].date()})")

    # 5. Process to weekly (identical to original logic)
    daily_df['precip_mm'] = daily_df['precip_m'] * 1000.0

    daily_df['year'] = daily_df['date'].dt.year
    daily_df['day_of_year'] = daily_df['date'].dt.dayofyear
    
    # Week number: days 1-7 = week 1, ..., days 358-364 = week 52, days 365+ = week 52
    daily_df['week'] = ((daily_df['day_of_year'] - 1) // 7 + 1).clip(upper=52)

    weekly = (daily_df.groupby(['year', 'week'])['precip_mm']
              .mean()
              .reset_index()
              .rename(columns={'precip_mm': 'precipitation_mm_per_day'}))

    # Ensure all year-week combinations exist
    full_index = pd.DataFrame(
        [(y, w) for y in years for w in range(1, 53)],
        columns=['year', 'week']
    )
    weekly = full_index.merge(weekly, on=['year', 'week'], how='left')
    assert not weekly['precipitation_mm_per_day'].isna().any()  # every day was checked above

    # 6. Save and print summary
    weekly.to_csv(output_csv, index=False)

    for year in years:
        yr_data = weekly[weekly['year'] == year]
        annual_total = yr_data['precipitation_mm_per_day'].sum() * 7
        print(f"  {subsite_name} {year}: annual total ~{annual_total:.1f} mm, "
              f"weekly range [{yr_data['precipitation_mm_per_day'].min():.2f}, "
              f"{yr_data['precipitation_mm_per_day'].max():.2f}] mm/day")

    print(f"  Weekly CSV saved to {output_csv}")


def main():
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    years = list(range(2013, 2024))  # 2013-2023

    # Find all subsites with AOI files
    subsites = []
    for subsite_dir in sorted(glob.glob(os.path.join(data_dir, 'subsite_*'))):
        name = os.path.basename(subsite_dir)
        aoi_path = None
        for aoi_name in ['aoi', 'aoi.txt']:
            candidate = os.path.join(subsite_dir, aoi_name)
            if os.path.exists(candidate):
                aoi_path = candidate
                break

        if aoi_path is None:
            print(f"Skipping {name}: no AOI file found")
            continue

        subsites.append((name, aoi_path, subsite_dir))

    print(f"Found {len(subsites)} subsites with AOI files")

    for subsite_name, aoi_path, subsite_dir in subsites:
        print(f"\n{'='*60}")
        print(f"Processing {subsite_name}")
        print(f"{'='*60}")

        bbox = parse_aoi(aoi_path)
        print(f"  Bounding box: {bbox}")

        precip_dir = os.path.join(subsite_dir, f'{subsite_name}_precip')
        os.makedirs(precip_dir, exist_ok=True)
        csv_path = os.path.join(precip_dir, f'{subsite_name}_weekly_precip.csv')

        if os.path.exists(csv_path):
            print(f"  CSV already exists, skipping: {csv_path}")
        else:
            fetch_and_process_weekly_csv(subsite_name, bbox, csv_path, years)

    print(f"\n{'='*60}")
    print("All subsites processed successfully!")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()