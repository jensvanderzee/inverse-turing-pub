#!/usr/bin/env julia
"""
How much discretisation error do the published fits carry at their own settings?

Motivation: the explicit scheme's step limit is set by the *combined* diffusion and
reaction rates (see [`spectral_bound`](@ref)), not by diffusion alone. Biomass runs
to the NDVI ceiling of 1500 g/m², so the reaction terms `r₂B` and `r₁B` are often
the binding constraint — and a parameter set can sit comfortably inside the
diffusion-only bound while being badly under-resolved.

Each fitted parameter set is integrated for one year at the settings it was fitted
with, and compared against an adaptive `ROCK2` reference solve of the same PDE.

Findings on the shipped 47-model table (site b, 1 year):

- 32 models are accurate: relative error ≤ 5e-5, stability ratio 1.06–1.08.
- 15 models are not: 11 exceed 1% error, 4 diverge to `NaN`. Stability ratio
  1.15–2.09. Worst case ~10%.
- The **diffusion-only** bound flags **none** of the 15.

Usage
-----
    julia --project=julia -t auto julia/scripts/accuracy_audit.jl

Options
    --params <csv>        parameter table to audit
    --site b              subsite supplying the initial field and forcing
    --steps-per-week 3    setting to audit (production fits used 3)
    --multiplier 1500     NDVI ceiling, used as the biomass scale in the bound
    --out <dir>
"""

using InverseTuring
using OrdinaryDiffEqStabilizedRK
using Printf
import Statistics, CSV, DataFrames

include(joinpath(@__DIR__, "common.jl"))

const PARAM_CSV = string(argval("params", joinpath(PY_RESULTS, "parameter_history_analysis",
                                                   "four_site_final_parameter_values.csv")))
const SITE = string(argval("site", "b"))
const SPW = argint("steps-per-week", 3)
const MULTIPLIER = argfloat("multiplier", 1500.0)
const OUTDIR = string(argval("out", joinpath(OUT_ROOT, "accuracy_audit")))

banner("Discretisation-error audit of fitted parameter sets")
report_threads()

ptab = read_parameter_table(PARAM_CSV)
series = load_site(DATA_DIR, SITE; multiplier = MULTIPLIER, T = Float64)
init = series.observations[1].biomass
weekly = series.observations[1].weekly_precipitation
cfg = SimConfig(steps_per_week = SPW, year_time_units = 1.0)
refcfg = SimConfig(steps_per_week = 4, year_time_units = 1.0)

@printf("models        : %d from %s\n", DataFrames.nrow(ptab), basename(PARAM_CSV))
@printf("audited at    : steps_per_week = %d  (dt = %.5f)\n", SPW, 1 / (52 * SPW))
@printf("biomass scale : %.0f g/m^2 (NDVI ceiling)\n", MULTIPLIER)
@printf("reference     : ROCK2, abstol 1e-4 / reltol 1e-8\n\n")

@printf("%-6s %8s %9s %8s %13s %13s %11s %9s\n",
        "model", "max d", "lambda", "ratio", "reference", "fixed-step", "rel error", "verdict")

rows = NamedTuple[]
for row in eachrow(ptab)
    p = params_from_row(row)
    d = maximum(paramvector(p)[1:3])
    lam = spectral_bound(p, MULTIPLIER)
    ratio = stability_ratio(p, cfg, MULTIPLIER)

    ref = try
        sol = InverseTuring.solve_year(p, pack_state(init), weekly, refcfg;
                                       alg = ROCK2(), abstol = 1e-4, reltol = 1e-8)
        Statistics.mean(biomass_of(sol.u[end]))
    catch e
        @warn "reference solve failed" model = row.model_id
        NaN
    end

    s = simstate(init)
    simulate_years!(s, p, weekly, cfg, 1)
    got = Statistics.mean(s.biomass)
    err = abs(got - ref) / ref

    verdict = !isfinite(err) ? "DIVERGED" : err > 0.01 ? "BAD" : err > 1e-3 ? "marginal" : "ok"
    @printf("%-6d %8.1f %9.1f %8.2f %13.5f %13s %11s %9s\n",
            row.model_id, d, lam, ratio, ref,
            isfinite(got) ? @sprintf("%.5f", got) : "NaN",
            isfinite(err) ? @sprintf("%.2e", err) : "-", verdict)
    push!(rows, (model_id = row.model_id, max_diffusion = d, spectral_bound = lam,
                 stability_ratio = ratio, reference = ref, fixed_step = got,
                 rel_error = err, verdict = verdict,
                 flagged_by_diffusion_only = d > diffusion_stability_limit(cfg)))
    flush(stdout)
end

df = DataFrames.DataFrame(rows)
bad = filter(r -> r.verdict in ("BAD", "DIVERGED"), eachrow(df))
good = filter(r -> r.verdict in ("ok", "marginal"), eachrow(df))

banner("Summary")
@printf("accurate (<=1%% error) : %3d models, stability ratio %.2f - %.2f\n",
        length(good), minimum(r -> r.stability_ratio, good), maximum(r -> r.stability_ratio, good))
if !isempty(bad)
    @printf("inaccurate or divergent: %3d models, stability ratio %.2f - %.2f\n",
            length(bad), minimum(r -> r.stability_ratio, bad), maximum(r -> r.stability_ratio, bad))
    @printf("  worst relative error : %.1f%%  (model %d)\n",
            100 * maximum(r -> isfinite(r.rel_error) ? r.rel_error : 0.0, bad),
            bad[argmax([isfinite(r.rel_error) ? r.rel_error : 0.0 for r in bad])].model_id)
    @printf("  diverged to NaN      : %d\n", count(r -> r.verdict == "DIVERGED", bad))
    @printf("  caught by the diffusion-only bound: %d of %d\n",
            count(r -> r.flagged_by_diffusion_only, bad), length(bad))
    println("\naffected models: ", join(sort([r.model_id for r in bad]), ", "))
    need = ceil(Int, SPW * maximum(r -> r.stability_ratio, bad) / 1.1)
    @printf("\nsteps_per_week needed to bring every model under the threshold: %d\n", need)
    println("Re-fitting those models at that setting would change their parameters;")
    println("this audit says nothing about which values are correct, only that the")
    println("flagged ones are not resolved at the discretisation they were fitted with.")
else
    println("every model is adequately resolved at this setting.")
end

mkpath(OUTDIR)
CSV.write(joinpath(OUTDIR, "accuracy_audit.csv"), df)
println("\nWritten to ", joinpath(OUTDIR, "accuracy_audit.csv"))
