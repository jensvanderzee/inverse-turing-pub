# Inverse Turing patterns

Code and data for learning the parameters of a Rietkerk-type dryland vegetation
model (an "inverse PDE") from satellite NDVI time series, and for comparing that
approach against a black-box recurrent convolutional network (RCNN).

The model has three coupled reaction–diffusion fields: surface water `O`, soil
water `W` and biomass `B`. Nine scalar coefficients are learned by
differentiating a multi-year rollout driven by weekly precipitation and fitting
the predicted year-on-year change in biomass to the observed change.

The pipeline is written in PyTorch. `julia/` holds a Julia port of it that has
been validated against the Python outputs (see [`julia/README.md`](julia/README.md)).

## Repository layout

| path | contents |
|:--|:--|
| `data/` | Input data per subsite (`subsite_a` … `subsite_k`): yearly NDVI GeoTIFFs for 2013–2022 (`*_ndvi/`), annual and weekly ERA5 precipitation (`*_precip/`), and the area-of-interest polygon (`aoi` / `aoi.txt`) |
| `*.py` (root) | Training, testing and analysis scripts, listed below |
| `*.sh` | SLURM job scripts used on the HPC cluster |
| `python_1site/` | Driver that fits the real-data model on a single site |
| `julia/` | `InverseTuring.jl`, the Julia port, with its own scripts and tests |

**`results/` is not included in this repository.** Every training script writes
to it, and the analysis and plotting scripts read from it, so run the
corresponding training step first. The manuscript and submission material are
also kept out of the repository.

## Sites

- **Training:** subsites `b`, `i`, `c`, `e`
- **Held-out testing:** subsites `f`, `k`, `j`
- **Initial conditions for forward simulations:** subsite `a`

The weekly precipitation CSVs for held-out subsites `k` (every year) and `j` (2020
onwards) are incomplete: they came from an Earth Engine collection that ends in July
2020, with the missing days written as zero rain. The loaders detect a weekly record
that delivers less than half the annual total and fall back to the annual total spread
evenly across 52 weeks. Re-fetch those two files with
`download_missing_weekly_precip.py` to replace the fallback with real weekly forcing.

## Scripts

Run all scripts from the repository root. Many are written as `#%%` cells
so they can also be stepped through interactively.

### Data preparation

| script | purpose |
|:--|:--|
| `download_era5_precip.py` | Download ERA5 daily precipitation through the Copernicus CDS API and compute weekly averages (needs `~/.cdsapirc`) |
| `download_era5_precip_gee.py` | Same as above, but through Google Earth Engine |
| `download_missing_weekly_precip.py` | Create AOI files and fetch weekly precipitation for the subsites that only had annual totals (`j`, `k`) |
| `view_site.py` | Quick look at the NDVI raster for one site and year |
| `plot_site_map.py` | Map of all subsite locations |

### Synthetic experiments

Parameters are recovered from data generated with known ground-truth values.

| script | purpose |
|:--|:--|
| `train_invPDE_synthetic_batch.py` | Train the inverse PDE on synthetic data from four sites (`--start`, `--end`, `--gpu`) |
| `train_invPDE_synthetic_batch_1site.py` | Same, for one site |
| `train_rcnn_batch.py` | Train the RCNN baseline on the same synthetic data |
| `compare_invPDE_synthetic_params.py` | Learned vs ground-truth parameters, four sites |
| `compare_invPDE_synthetic_params_1site.py` | Learned vs ground-truth parameters, one site |
| `compare_1site_vs_4site_params.py`, `compare_1site_vs_4site_relerr.py` | Parameter values and relative errors, one site vs four sites |
| `synthetic_dataregime_analysis.py` | Effect of adding more training sites |
| `synthetic_sensitivity_analysis.py` | One-at-a-time sensitivity of simulated biomass to each parameter |
| `rcnn_vs_PDE_synthetic.py` | Extrapolation ability of the RCNN vs the inverse PDE |
| `rcnn_extrapolation_test.py` | Best RCNN and inverse-PDE models tested on longer horizons and out-of-range precipitation |
| `invPDE_1site_extrapolation.py` | Extrapolation test for the one-site inverse PDE |
| `rcnn_simulate.py` | Forward simulation with the best RCNN model |

### Real satellite data

| script | purpose |
|:--|:--|
| `realdata_train_invPDE.py` | Fit PDE parameters to the four training sites (7500 epochs per model) |
| `realdata_train_invPDE_10to20.py` | Same, for model indices 10–19, so runs can be split across jobs |
| `find_stable_seed.py` | Find a seed that trains stably for biomass multipliers 750, 1500 and 3000 |
| `multiplier_check_analysis.py` | Compare runs with different biomass multipliers |
| `realdata_test_invPDE.py` | Score the fitted models on the held-out sites |
| `realdata_parameter_analysis.py` | Agreement of learned parameters across runs |
| `realdata_parameter_correlations.py` | Correlations and trade-offs between learned parameters |
| `compare_invPDE_realdata_params.py` | Parameter trajectories across all real-data runs |
| `compare_invPDE_realdata_params_with_composite.py` | Same, with a composite figure |
| `realdata_simulate_invPDE.py` | Forward simulation with the best tested model |
| `bifurcation_parallel.py` | Equilibrium biomass vs annual precipitation for every model (multiprocessing) |
| `realdata_bifurcation_plot.py` | Serial version of the bifurcation sweep and plot |
| `plot_bifurcation.py`, `plot_bifurcation_cosmetic.py`, `bifurcation_plot_only.py` | Redraw the bifurcation figure from saved sweep data |
| `python_1site/train_1site_realdata.py` | Fit the real-data model on one site (`--site b`) for comparison with the four-site fit |

### Numerics and runtime

| script | purpose |
|:--|:--|
| `compare_forcing_frequency.py` | Same annual rainfall delivered daily, weekly or as one annual pulse (`compare_forcing_frequency_nb.py` is the cell-by-cell version) |
| `measure_runtime.py` | Measure seconds per epoch for each experiment and extrapolate to a full run; writes `runtime_estimates.csv` and `runtime_estimates.md` |
| `measure_runtime_colab.ipynb` | Notebook for running `measure_runtime.py` on Colab |

Estimated time for one training run on an NVIDIA A100 (from
[`runtime_estimates.md`](runtime_estimates.md)):

| experiment | epochs | estimated wall time |
|:--|--:|--:|
| invPDE synthetic, 1 site | 7500 | 2 h |
| invPDE synthetic, 4 sites | 7500 | 8 h |
| RCNN synthetic, 4 sites | 1000 | 39 min |
| invPDE real data, 4 sites | 7500 | 25 h |

## Requirements

Python 3.12 with PyTorch, plus `numpy`, `pandas`, `matplotlib`, `seaborn`,
`scipy`, `scikit-learn`, `rasterio`, `tqdm` and `Pillow`. Some scripts need extra
packages:

- `plot_site_map.py`: `contextily`, `pyproj`
- `download_era5_precip.py`: `cdsapi`, `xarray`, `netcdf4`
- `download_*_gee.py`, `download_missing_weekly_precip.py`: `earthengine-api`

A GPU is recommended for training. For the Julia port, see
[`julia/README.md`](julia/README.md).

## Running on the cluster

The `*.sh` files are SLURM job scripts, each requesting one GPU for up to 48 hours.
They call the Python scripts by absolute path on the cluster
(`/home/WUR/zee034/inverse-turing-testing/...`), so change that path before
using them elsewhere.

## License

MIT, see [`LICENSE`](LICENSE).
