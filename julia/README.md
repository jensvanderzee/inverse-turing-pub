# InverseTuring.jl

Julia port of the inverse-PDE pipeline that fits a Rietkerk-type dryland vegetation
model to satellite NDVI time series.

The model is three coupled reaction–diffusion fields — surface water `O`, soil
water `W`, biomass `B` — integrated with explicit Euler on a 5-point Laplacian:

```
∂ₜO = d₁ ∇²O − l₁ O + R − r₂ O B
∂ₜW = d₂ ∇²W − l₂ W + r₂ O B − r₁ W B
∂ₜB = d₃ ∇²B − l₃ B + j r₁ W B
```

Nine scalar coefficients are recovered by differentiating a multi-year rollout and
descending on the mismatch between predicted and observed year-on-year change in
biomass.

## Status

The port reproduces the PyTorch implementation. `scripts/validate_against_python.jl`
runs **105 checks against artefacts the Python code left in `results/`** — nothing
is re-fitted, so agreement means the two implementations agree on the data
pipeline, the discretisation, and the metrics:

| check | agreement |
|:--|:--|
| NDVI → biomass statistics vs `data_info.json` | min/max exact, mean to 5e-9 |
| Turing diagnostic, all 47 models | exact |
| 9-year rollout on 3 held-out sites (1 872 Euler steps, Float32) | ~1e-8 relative |
| Full 141-entry `test_metrics.csv` | median 2e-8, same best model (42), same divergent case |
| All 22 statistics in `four_site_agreement_table.csv` | exact to 4 published decimals |
| 1000-year equilibrium rollouts vs `bifurcation_data.csv` (208 000 steps) | 2.5e-7 on the vegetated branch |

The remaining differences are float32 round-off, not disagreement about the model.
Pass `--full` to widen the last check, which also covers the collapsed branch —
there biomass decays through nine orders of magnitude and the two still agree to
1.2e-4.

`Pkg.test()` runs 221 unit tests covering the operator, the loss, AD against finite
differences, the optimiser, the stability bound, IO round-trips, and the reference
agreement above.

## Performance

`scripts/benchmark.jl` times the Julia fit the same way `measure_runtime.py` timed
the PyTorch one, so the two are directly comparable. Julia figures below are
best-of-4 epochs on 8 threads of a 32-core desktop; PyTorch figures are the
recorded A100 numbers from `../runtime_estimates.csv`, at the same
`steps_per_week`.

| experiment | PyTorch, A100 | Julia, CPU | |
|:--|--:|--:|--:|
| invPDE real data, 4 sites | 12.10 s/epoch | 4.34 s/epoch | **2.8×** |
| invPDE synthetic, 4 sites | 3.80 s/epoch | 0.86 s/epoch | **4.4×** |
| invPDE synthetic, 1 site | 0.97 s/epoch | 0.81 s/epoch | 1.2× |
| → full 7500-epoch real-data fit | 25 h | 9 h | |

This is CPU-Julia against GPU-PyTorch, so it says nothing about language speed in
isolation. What it shows is that this particular problem is a bad fit for a GPU:

- **A 131×140 grid is too small.** One loss evaluation is ~20 000 sequential tiny
  kernel launches, and launch overhead dominates the arithmetic. The margin is
  smallest in the 1-site synthetic case — the one with the least per-epoch
  dispatch — which is exactly what that explanation predicts.
- **Forward-mode AD beats reverse mode at nine parameters.** One `ForwardDiff`
  pass costs about ten primal evaluations but tapes nothing; reverse mode would
  have to store thousands of sequential Euler steps.

Two tuning results worth knowing:

- The ForwardDiff chunk width defaults to **5, not 9**. Taking all nine partials in
  one pass looks optimal — one pass instead of two — but is **2.5× slower**,
  because a `Dual{9,Float64}` field is 80 bytes per cell and the working set falls
  out of L2. See `DEFAULT_CHUNK` and `autotune_chunk`.
- Threading a single fit saturates at about **4 threads**, because there are only
  four sites and the transitions within a site are sequential. Beyond that, use
  `train_many(...; parallel = :runs)` to fit several restarts at once — that is
  where the remaining cores go.

## Install

Julia ≥ 1.10.

```bash
julia --project=julia -e 'using Pkg; Pkg.instantiate()'
```

Pass `-t auto` (or at least `-t 8`); the fits and sweeps are threaded and the
difference is roughly 3×.

## Quick start

```bash
julia --project=julia -t auto julia/scripts/validate_against_python.jl
```

```julia
using InverseTuring

sites = load_sites("data", ["b", "i", "c", "e"])
cfg   = SimConfig(steps_per_week = 3, year_time_units = 1.0)
prob  = InverseProblem(sites, cfg)

result = train(prob; cfg = TrainConfig(epochs = 7500), seed = 77)
show(stdout, MIME"text/plain"(), result.params)
turing_value(result.params) < 0    # true => vegetation patterns are possible
```

## Scripts

Each mirrors a Python entry point. All take `--help`-style options documented in
their header docstring.

| script | replaces | what it does |
|:--|:--|:--|
| `train_realdata.jl` | `realdata_train_invPDE.py` | fit coefficients to satellite data |
| `train_synthetic.jl` | `train_invPDE_synthetic_batch{,_1site}.py` | recover known parameters from synthetic data |
| `test_realdata.jl` | `realdata_test_invPDE.py` | score fitted models on held-out sites |
| `bifurcation.jl` | `bifurcation_parallel.py` | equilibrium biomass vs rainfall |
| `extrapolation.jl` | `invPDE_1site_extrapolation.py` | 1000-year rollout outside the training envelope |
| `parameter_analysis.jl` | `realdata_parameter_analysis.py` | cross-run agreement statistics |
| `benchmark.jl` | `measure_runtime.py` | timings vs the recorded A100 run |
| `validate_against_python.jl` | — | equivalence checks against `results/` |
| `accuracy_audit.jl` | — | discretisation error carried by each fitted parameter set |
| `diffeq_comparison.jl` | — | fixed-step sweep vs adaptive ODE solvers |

Example:

```bash
julia --project=julia -t auto julia/scripts/train_realdata.jl --models 10 --epochs 7500 --steps-per-week 3
```

### Where things are read and written

The existing `results/` directory is treated as **read-only**. Scripts load
parameter tables, metrics and histories from it, but every output goes to
`julia/results/` — so a Julia run can never overwrite a published figure or table,
and the port is self-contained. Override per run with `--out`.

## Interoperability

The port reads and writes the Python formats, so the two codebases can share a
`results/` directory:

- **Parameter histories**: `load_parameter_history` reads Python `.pkl`
  (via Pickle.jl) and Julia `.json`.
- **Parameter tables**: `read_parameter_table` / `write_parameter_table` use the
  same CSV columns as `four_site_final_parameter_values.csv`.
- **Run records**: `save_run` matches the Python `result_dict` schema.
- **Rasters**: `ndvi_biomass` transposes GDAL's `(x, y)` layout to `(row, col)`,
  so fields have the same orientation as `rasterio` and every figure matches.

## Things worth knowing

These are properties of the original model that the port makes explicit rather
than changes.

**The field update is sequential, not simultaneous.** `step!` advances `O`, then
uses the *updated* `O` for `W`, then the *updated* `W` for `B` — a Gauss–Seidel
sweep, not a forward Euler step. This is what the Python code does (each
assignment rebinds the name before the next expression is built), and it looks
incidental rather than intended. It means the scheme is not a method-of-lines
discretisation of an ODE, so it cannot be handed to an ODE solver as-is.

Measured on the published best-fit parameters at 4 steps/week, the two orders
differ by ~1e-6 relative in nine-year delta-MSE on well-behaved sites and ~5e-4 on
the stiffest one. That is far below the scatter between random restarts, so no
scientific conclusion depends on it — but it is above the 1e-8 agreement this port
achieves against the Python outputs, so reproducing *those numbers* does.

**The explicit scheme's step limit is set by diffusion *and* reaction — and 15 of
the 47 published fits are under-resolved at their own settings.** This is the most
consequential thing the port turned up; `scripts/accuracy_audit.jl` reproduces it.

Linearising, the fastest decay rate is `max(8d₁+l₁+r₂B, 8d₂+l₂+r₁B, 8d₃+l₃)`, and
explicit Euler needs `dt·λ ≤ 2`. Because biomass runs to the NDVI ceiling of
1500 g/m², the reaction terms `r₂B`/`r₁B` are often larger than the diffusive `8d`.

Each fitted parameter set integrated for one year at `steps_per_week = 3` (what
production used) against an adaptive ROCK2 reference:

| | count | `stability_ratio` | relative error |
|:--|--:|:--|:--|
| well resolved | 32 | 1.26 – 1.45 | ≤ 1% |
| under-resolved | 11 | 2.72 – 5.11 | 1% – **10.3%** |
| divergent (`NaN`) | 4 | 2.72 – 5.11 | — |

The groups separate cleanly with nothing between 1.45 and 2.72. Affected models:
**3, 5, 13, 17, 43, 44, 49, 51, 55, 57, 65, 77, 92, 97, 98**. They all have *small*
diffusion coefficients and large reaction rates — so `diffusion_stability_limit`
alone flags **none** of them. Use `stability_ratio(p, cfg, 1500)` or
`check_stability(p, cfg, 1500)`; `steps_per_week = 14` would bring every model
under the threshold.

This does not say which parameter values are right — only that the flagged ones are
not converged at the discretisation they were fitted with, so any conclusion resting
on them deserves a re-fit at a finer step.

**The loss is fitted on differences, not absolute fields.** `delta_loss = true`
(the default, `use_delta_loss` in Python) scores the year-on-year *change* in
biomass. This removes each site's static spatial mean from the objective, so the
parameters are constrained by how the vegetation moves rather than by how much of
it there is. All published fits use it; `delta_loss = false` gives the absolute
form for comparison.

**`year_time_units` differs between experiments.** The real-data and 1-site
synthetic scripts use `1.0`; the 4-site synthetic script uses `1.5`. Fitted rate
constants from the two are not comparable, so it is an explicit field of
`SimConfig` rather than a hard-coded constant.

**`summer_weekly_precip` under-delivers by 0.27 %.** It normalises with `365/52 ≈
7.019` days per week while the integrator converts rates to volumes with exactly
`7`. The quirk is inherited from `bifurcation_parallel.py` and kept so that a
rainfall value on a Julia bifurcation diagram means what it does on the published
one. Pass `days_per_week = 7` to drop it.

**Two `parameters/` directories hold different runs under identical filenames.**
`realdata_parameter_analysis.py` reads `results/real_data/parameters/`;
`bifurcation_parallel.py` reads `results/real_data/models/parameters/`. They
disagree — model 82 is a diverged 3-snapshot run in the first and a healthy
751-snapshot run in the second. The scripts here default to the same directory as
the Python script they replace, and say so. **This one is a latent hazard in the
project rather than a quirk of the port**; worth consolidating.

**The two Python analyses also filter runs differently, and it changes a
conclusion.** `tier1_filter` (history length *and* degenerate values, used by the
bifurcation script) keeps 32 runs; `drop_degenerate` (values only, used by the
parameter analysis) keeps 47. On the 32-run set the composite growth efficiency is
far better constrained than its individual rate constants — CV 0.03 against
0.10–0.22, the signature of a problem that identifies a combination rather than the
parameters. On the 47-run set it is not (CV 1.63 against 0.64–1.14), because that
set still contains fits that landed in entirely different regimes. Both filters are
provided, `parameter_analysis.jl` takes `--filter`, and it prints which one it used.
**Quote the filter when reporting this.**

## The adaptive ODE backend

`using OrdinaryDiffEqStabilizedRK` activates an extension that solves the
*continuous* PDE by method of lines, as an alternative to the default fixed-step
sweep. `solve_year`, `simulate_years_ode` and `ode_loss` become available;
`using InverseTuring` alone stays at ~3.5 s and does not load it.

```julia
using InverseTuring, OrdinaryDiffEqStabilizedRK
sol = solve_year(p, pack_state(biomass0), weekly, cfg; alg = ROCK2(),
                 abstol = 1e-3, reltol = 1e-6)
```

**Set `abstol` to the scale of the data.** Biomass is in g/m² and runs to ~10³, so
the SciML default of `1e-6` — or `1e-8` "to be safe" — demands eleven or twelve
significant digits and costs ~100× more work than the data can justify. Pair a
loose `abstol` with a tight `reltol`.

### Is it worth switching to? Mostly no — measured, not assumed

`scripts/diffeq_comparison.jl`, one year on the real grid:

| backend | rel. error | steps | f-evals | time |
|:--|--:|--:|--:|--:|
| fixed-step, 4/week | 1.4e-05 | 208 | 208 | **0.04 s** |
| fixed-step, 16/week | 2.2e-06 | 832 | 832 | 0.15 s |
| ROCK2, abstol 1e-4 / reltol 1e-8 | 1.6e-05 | 2208 | 11103 | 2.81 s |
| Tsit5, abstol 1e-3 / reltol 1e-6 | 5.1e-05 | 345 | 3027 | 0.84 s |

**~21× slower at matched accuracy.** The reason is that the problem is *not stiff
at the step sizes in use*: the diffusive time scale is 1/(8·d_max) ≈ 0.0037, so
stability alone needs ~270 steps per year and the production setting already takes
208. Accuracy and stability bite at the same step size — precisely the regime where
adaptivity and stabilised/implicit methods have nothing to recover. (Tsit5, a
*non-stiff* method, beating ROCK2 is the tell.)

Implicit methods are worse than useless here: at 55 020 unknowns a dense Jacobian
would be ~24 TB.

**What the ODE backend does win is robustness.** Past the explicit bound the fixed
scheme returns `NaN` and the fitting run is lost; the adaptive solver shortens its
step and continues:

| diffusion scale | max d | fixed-step, 4/week | ROCK2 |
|--:|--:|:--|:--|
| 1.0 | 33.7 | 131.9158 | 131.9113 |
| 1.5 | 50.6 | 125.1483 *(5% off, still "within" the bound)* | 131.9328 |
| 2.0 | 67.4 | **DIVERGED** | 131.9472 |
| 4.0 | 134.8 | **DIVERGED** | 131.9775 |

Gradients work: `ForwardDiff` propagates through an adaptive solve with no adjoint
machinery, agreeing with the fixed-step gradient to ~1e-3 — but ~119× slower.

### Fitting with it is not feasible — settled

| | fixed-step | ODE backend |
|:--|--:|--:|
| gradient, 1 site / 2 transitions | 0.89 s | 105.8 s |
| gradient, 4 sites / 36 transitions (extrapolated) | ~4.3 s | **~32 min** |
| 7500-epoch fit | ~9 h | **~166 days** |

The gap is structural, not a tuning problem. The fixed scheme costs **one** RHS
evaluation per step with no error estimate, no rejections and no controller; an
adaptive solver spends 5–10 evaluations per accepted step and cannot take longer
steps, because the problem is not stiff at the step size in use. Nothing in the
solver menu recovers two orders of magnitude.

Caveat on scope: only `ForwardDiff`-through-the-solver was tested, not
`SciMLSensitivity` adjoints. With nine parameters forward mode is normally the
better choice, and an adjoint still has to integrate adaptively forwards and
backwards, so it would plausibly save a small factor — not the ~100× needed. If
someone wants to settle that too, `lossfn` in [`train`](@ref) is the hook.

**Verdict:** keep the fixed-step backend for all fitting. The ODE backend stays
useful for what it has already delivered — a discretisation-independent check that
found 15 under-resolved fits (see above) — plus forward runs outside the fitted
parameter range. Fitted parameters are not transferable between the two
discretisations in any case.

## Element type

`Float64` by default, which is the better base for gradient-based fitting.
`Float32` reproduces the original PyTorch arithmetic bit-for-bit and is what the
validation script uses. Array type is generic too: `laplacian!` has a
scalar-indexing-free fallback, so a GPU array works by construction — but only the
CPU path is tested here, and given the grid size a GPU is unlikely to help.

## Layout

```
src/
  parameters.jl   RietkerkParams, presets, conversions
  operators.jl    laplacian! (replicate boundaries)
  model.jl        step!, simulate_year!, SimConfig, stability limit
  forcing.jl      weekly precipitation profiles
  realdata.jl     GeoTIFF NDVI + ERA5 precipitation loading
  problem.jl      InverseProblem, loss, evaluate
  synthetic.jl    ground-truth data generation
  optim.jl        PyTorch-compatible Adam
  train.jl        train, train_many, TrainConfig
  analysis.jl     Turing/composite diagnostics, filters, bifurcation sweep
  io.jl           JSON/CSV/pickle interoperability
```

## Not ported

Listed explicitly so the gap is visible rather than implied.

**Deliberately out of scope**

- *Data acquisition* — `download_era5_precip.py`, `download_era5_precip_gee.py`,
  `download_missing_weekly_precip.py`. These depend on `cdsapi` and
  `earthengine-api`, which are Python-only. Keep using them; they write the CSVs
  this package reads.
- *RCNN baseline* — `train_rcnn_batch.py`, `rcnn_simulate.py`,
  `rcnn_extrapolation_test.py`, `rcnn_vs_PDE_synthetic.py`. A neural-network
  comparison model, not part of the inverse-PDE pipeline.

**Analyses still on the Python side.** Each is a short script on top of what the
package already exposes — none needs new numerics — but none has been written yet:

| Python script | what it would need |
|:--|:--|
| `synthetic_sensitivity_analysis.py` | OAT sensitivity sweep over `RietkerkParams` |
| `synthetic_dataregime_analysis.py` | refit with `n_sites = 1…4`, compare recovery |
| `realdata_parameter_correlations.py` | correlations over a parameter table |
| `realdata_simulate_invPDE.py` | forward rollout — `simulate_years!` covers it |
| `multiplier_check_analysis.py`, `find_stable_seed.py` | NDVI-multiplier sensitivity |
| `compare_*.py` (6 scripts), `plot_site_map.py`, `view_site.py`, `plot_bifurcation_cosmetic.py` | figure formatting only |

The Python versions of these still run against `results/`, and the Julia scripts
write the same formats, so the two can be mixed while the rest is ported.
