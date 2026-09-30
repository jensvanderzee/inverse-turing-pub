# -*- coding: utf-8 -*-
"""
Download ERA5 daily total precipitation for each subsite and compute weekly averages.

Run in the 'dataio' conda environment:
    conda activate dataio
    python download_era5_precip.py

Prerequisites:
    1. Install dependencies:
        conda install -c conda-forge cdsapi xarray netcdf4 pandas numpy

    2. Set up CDS API credentials:
        - Create an account at https://cds.climate.copernicus.eu/
        - Go to your profile page and copy your API key
        - Create a file at ~/.cdsapirc (Linux/Mac) or C:\\Users\\<username>\\.cdsapirc (Windows)
          with the following content:

            url: https://cds.climate.copernicus.eu/api
            key: <your-uid>:<your-api-key>

        For the new CDS-Beta API (2024+), the format may be:
            url: https://cds-beta.climate.copernicus.eu/api
            key: <your-personal-access-token>
"""

import os
import re
import glob
import cdsapi
import xarray as xr
import pandas as pd
import numpy as np
from pathlib import Path


def parse_aoi(aoi_path: str) -> dict:
    """
    Parse an AOI file containing a bounding box as coordinate pairs.
    Format: [lon1,lat1],[lon2,lat2],...

    Returns dict with keys: north, south, east, west
    (CDS API area format)
    """
    with open(aoi_path, 'r') as f:
        content = f.read().strip()

    # Extract all [lon,lat] pairs
    pairs = re.findall(r'\[([-\d.]+),([-\d.]+)\]', content)
    if not pairs:
        raise ValueError(f"Could not parse coordinates from {aoi_path}: {content}")

    lons = [float(p[0]) for p in pairs]
    lats = [float(p[1]) for p in pairs]

    return {
        'north': max(lats),
        'south': min(lats),
        'east': max(lons),
        'west': min(lons),
    }


def download_era5_daily(subsite_name: str, bbox: dict, output_nc: str,
                        years: list):
    """
    Download ERA5 daily total precipitation for a subsite's bounding box.

    ERA5 'total_precipitation' is accumulated over the forecast step (in meters).
    We request daily data which gives the daily accumulated total.
    """
    client = cdsapi.Client()

    # CDS API area format: [north, west, south, east]
    area = [bbox['north'], bbox['west'], bbox['south'], bbox['east']]

    print(f"  Downloading ERA5 for {subsite_name}")
    print(f"  Bounding box (N,W,S,E): {area}")
    print(f"  Years: {years[0]}-{years[-1]}")

    client.retrieve(
        'reanalysis-era5-single-levels',
        {
            'product_type': 'reanalysis',
            'variable': 'total_precipitation',
            'year': [str(y) for y in years],
            'month': [f'{m:02d}' for m in range(1, 13)],
            'day': [f'{d:02d}' for d in range(1, 32)],
            'time': '00:00',
            'area': area,
            'format': 'netcdf',
        },
        output_nc
    )
    print(f"  Downloaded to {output_nc}")


def process_to_weekly_csv(nc_path: str, output_csv: str, subsite_name: str):
    """
    Read ERA5 daily NetCDF, spatially average over the domain, and compute
    weekly average precipitation rates (mm/day).

    Output CSV columns: year, week, precipitation_mm_per_day
    52 weeks per year (days 1-364 in 7-day bins, remaining days in week 52)
    """
    ds = xr.open_dataset(nc_path)

    # ERA5 variable name is 'tp' (total_precipitation) in meters
    tp_var = 'tp'
    if tp_var not in ds:
        # Try alternate names
        candidates = [v for v in ds.data_vars if 'precip' in v.lower() or v == 'tp']
        if candidates:
            tp_var = candidates[0]
        else:
            raise KeyError(f"No precipitation variable found in {nc_path}. "
                          f"Variables: {list(ds.data_vars)}")

    # Spatial average: mean over lat/lon dimensions
    spatial_dims = [d for d in ds[tp_var].dims if d not in ('time', 'valid_time')]
    daily_precip = ds[tp_var].mean(dim=spatial_dims)

    # Convert to pandas Series with datetime index
    time_dim = 'valid_time' if 'valid_time' in daily_precip.dims else 'time'
    daily_df = daily_precip.to_dataframe().reset_index()
    daily_df = daily_df.rename(columns={time_dim: 'date', tp_var: 'precip_m'})
    daily_df['date'] = pd.to_datetime(daily_df['date'])
    daily_df = daily_df.sort_values('date').reset_index(drop=True)

    # Convert meters to mm
    daily_df['precip_mm'] = daily_df['precip_m'] * 1000.0

    # Assign year and week (1-52)
    daily_df['year'] = daily_df['date'].dt.year
    daily_df['day_of_year'] = daily_df['date'].dt.dayofyear
    # Week number: days 1-7 = week 1, ..., days 358-364 = week 52, days 365+ = week 52
    daily_df['week'] = ((daily_df['day_of_year'] - 1) // 7 + 1).clip(upper=52)

    # Compute weekly mean precipitation rate (mm/day)
    weekly = (daily_df.groupby(['year', 'week'])['precip_mm']
              .mean()
              .reset_index()
              .rename(columns={'precip_mm': 'precipitation_mm_per_day'}))

    # Ensure all year-week combinations exist (fill missing with 0)
    years = sorted(weekly['year'].unique())
    full_index = pd.DataFrame(
        [(y, w) for y in years for w in range(1, 53)],
        columns=['year', 'week']
    )
    weekly = full_index.merge(weekly, on=['year', 'week'], how='left')
    weekly['precipitation_mm_per_day'] = weekly['precipitation_mm_per_day'].fillna(0.0)

    weekly.to_csv(output_csv, index=False)

    # Print summary
    for year in years:
        yr_data = weekly[weekly['year'] == year]
        annual_total = yr_data['precipitation_mm_per_day'].sum() * 7  # approximate annual mm
        print(f"  {subsite_name} {year}: annual total ~{annual_total:.1f} mm, "
              f"weekly range [{yr_data['precipitation_mm_per_day'].min():.2f}, "
              f"{yr_data['precipitation_mm_per_day'].max():.2f}] mm/day")

    ds.close()
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

        # Parse bounding box
        bbox = parse_aoi(aoi_path)
        print(f"  Bounding box: {bbox}")

        # Paths
        precip_dir = os.path.join(subsite_dir, f'{subsite_name}_precip')
        os.makedirs(precip_dir, exist_ok=True)
        nc_path = os.path.join(precip_dir, 'era5_daily_precip.nc')
        csv_path = os.path.join(precip_dir, f'{subsite_name}_weekly_precip.csv')

        # Download (skip if already exists)
        if os.path.exists(nc_path):
            print(f"  NetCDF already exists, skipping download: {nc_path}")
        else:
            download_era5_daily(subsite_name, bbox, nc_path, years)

        # Process to weekly CSV
        process_to_weekly_csv(nc_path, csv_path, subsite_name)

    print(f"\n{'='*60}")
    print("All subsites processed successfully!")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
