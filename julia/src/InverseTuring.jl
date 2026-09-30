"""
    InverseTuring

Julia port of the inverse-PDE pipeline that fits a Rietkerk-type dryland
vegetation model to satellite NDVI time series.

The model is three coupled reaction–diffusion fields — surface water, soil water,
biomass — integrated with explicit Euler on a 5-point Laplacian. Nine scalar
ecological coefficients are recovered by differentiating the whole multi-year
rollout and descending on the mismatch between predicted and observed year-on-year
change in biomass.

Layout:

| file             | contents                                                  |
|:-----------------|:----------------------------------------------------------|
| `parameters.jl`  | [`RietkerkParams`](@ref), presets, vector/dict conversions |
| `operators.jl`   | [`laplacian!`](@ref) with replicate boundaries             |
| `model.jl`       | [`step!`](@ref), [`simulate_year!`](@ref), [`SimConfig`](@ref) |
| `forcing.jl`     | weekly precipitation profiles                              |
| `realdata.jl`    | GeoTIFF NDVI + ERA5 precipitation loading                  |
| `problem.jl`     | [`InverseProblem`](@ref), [`loss`](@ref), [`evaluate`](@ref) |
| `synthetic.jl`   | ground-truth data generation                               |
| `optim.jl`       | PyTorch-compatible Adam                                    |
| `train.jl`       | [`train`](@ref) and its configuration                      |
| `analysis.jl`    | Turing/composite diagnostics, filtering, bifurcation sweep |
| `io.jl`          | JSON/CSV/pickle interoperability with the Python results   |

Element type is a free choice: `Float64` (default) is the better base for fitting,
`Float32` reproduces the original PyTorch arithmetic. Array type is free too — the
kernels have a scalar-indexing-free fallback, so a GPU array works by construction,
though only the CPU path is exercised by the test suite.
"""
module InverseTuring

using Printf: @printf, @sprintf
import Random
import Statistics

import ForwardDiff
import DiffResults
import ArchGDAL
import CSV
import DataFrames
import JSON
import Pickle

export RietkerkParams, PARAM_NAMES, NPARAMS, paramvector, paramdict, randparams,
       SYNTHETIC_TRUTH, SYNTHETIC_TRUTH_1SITE, REALDATA_REFERENCE
export laplacian, laplacian!
export SimConfig, SimState, simstate, copystate, step!, simulate_year!, simulate_years!,
       diffusion_stability_limit, spectral_bound, max_stable_dt, stability_ratio,
       check_stability
export sinusoidal_weekly_precip, summer_weekly_precip, uniform_weekly_precip, annual_total
export ndvi_biomass, YearObservation, SiteSeries, load_site, load_sites, biomass_stats,
       read_annual_precip, read_weekly_precip, years
export SiteTrajectory, InverseProblem, loss, trajectory_loss, evaluate, ntransitions,
       rethread, mean_squared_delta_error, mean_squared_error
export equilibrium_biomass, synthetic_series, synthetic_experiment
export AdamState, adam_step!, clip_global_norm!, decay_lr!
export TrainConfig, SYNTHETIC_TRAIN_CONFIG, TrainResult, train, train_many,
       loss_and_gradient, DEFAULT_CHUNK, autotune_chunk
export turing_value, composite_value, mean_training_biomass, tier1_filter,
       drop_degenerate, agreement_table, bifurcation_sweep
export load_parameter_history, load_parameter_histories, read_parameter_table,
       params_from_row, write_parameter_table, save_run, load_run, save_json
export WeeklyForcing, week_boundaries, rietkerk_rhs!, pack_state, unpack_state,
       biomass_of, solve_year, simulate_years_ode, ode_loss

include("parameters.jl")
include("operators.jl")
include("model.jl")
include("forcing.jl")
include("realdata.jl")
include("problem.jl")
include("synthetic.jl")
include("optim.jl")
include("train.jl")
include("analysis.jl")
include("io.jl")
include("diffeq.jl")

end # module
