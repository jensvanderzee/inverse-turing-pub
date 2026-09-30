#!/usr/bin/env julia
"""
Identifiability analysis of a hybrid Adam -> L-BFGS ensemble.

The question this answers: given many fits that describe the data equally well,
which of the nine coefficients does the data actually determine?

Filtering matters and is done explicitly. Two screens are applied and reported
separately, because pooling everything gives statistics that say more about the
failures than about the identifiable structure — the run's own progress reports
put mortality CV at 0.78 across all 30 models and at 0.06 across the fits that
converged, and only the second number is about identifiability.

    loss    < LOSS_CUT   the fit reached the basin at all
    ‖∇L‖    < GRAD_CUT   it is near a stationary point, not stopped mid-descent
                         by the evaluation cap

Bound-pinning is counted, not filtered: a coefficient sitting exactly on the
1e-4 clamp is the degenerate solution the project's `drop_degenerate` screen
rejects, and how often the optimiser lands there is part of the result.

Usage
-----
    julia --project=julia julia/scripts/analyse_hybrid_ensemble.jl [rundir] [--compare dir]
"""

using InverseTuring
using Printf
import CSV, DataFrames, Statistics

include(joinpath(@__DIR__, "common.jl"))

const RUNDIR = length(ARGS) >= 1 && !startswith(ARGS[1], "--") ? ARGS[1] :
    joinpath(OUT_ROOT, "hybrid_realdata")
const COMPARE = let i = findfirst(==("--compare"), ARGS)
    i === nothing ? nothing : ARGS[i + 1]
end
const LOSS_CUT = 743.0
const GRAD_CUT = 1e3
const FLOOR = 1.1e-4
const CFG = SimConfig(steps_per_week = 3, year_time_units = 1.0)
const BMAX = 1500.0

load(dir) = CSV.read(joinpath(dir, "progress.csv"), DataFrames.DataFrame)
pinned_names(r) = [n for n in PARAM_NAMES if r[n] <= FLOOR]
npinned(r) = length(pinned_names(r))
ratio(r) = stability_ratio(RietkerkParams([r[n] for n in PARAM_NAMES]), CFG, BMAX)
cv(v) = Statistics.std(v) / abs(Statistics.mean(v))

df = load(RUNDIR)
banner("Ensemble: $(DataFrames.nrow(df)) models")

# ---------------------------------------------------------------------------
short = Dict(zip(PARAM_NAMES, ["d1", "d2", "d3", "l1", "l2", "l3", "r2", "r1", "j"]))
@printf("%-6s %-9s %-10s %-11s %6s %9s  %s\n",
        "seed", "loss", "|grad|", "stage", "evals", "dt/dtmax", "pinned")
println("-" ^ 96)
for r in eachrow(DataFrames.sort(df, :hybrid_loss))
    @printf("%-6d %-9.2f %-10.3g %-11s %6d %9.2f  %s\n",
            r.seed, r.hybrid_loss, r.hybrid_gnorm,
            isnan(r.adam_loss) ? "lbfgs-only" : "hybrid", r.lbfgs_evals, ratio(r),
            join((short[n] for n in pinned_names(r)), " "))
end

# ---------------------------------------------------------------------------
"""Per-parameter spread over `sub`, plus the two candidate combinations."""
function spread(sub; label)
    n = DataFrames.nrow(sub)
    @printf("\n%s  (n=%d, loss %.2f-%.2f, spread %.2f%%)\n", label, n,
            extrema(sub.hybrid_loss)...,
            100 * (maximum(sub.hybrid_loss) / minimum(sub.hybrid_loss) - 1))
    @printf("%-30s %10s %8s %12s\n", "parameter", "median", "CV", "max/min")
    rows = [(n, sub[!, n]) for n in PARAM_NAMES]
    for (nm, v) in sort(rows, by = x -> cv(x[2]))
        @printf("%-30s %10.4g %8.3f %12.4g\n", nm, Statistics.median(v), cv(v),
                maximum(v) / minimum(v))
    end
    d12 = sub.surface_water_diffusion_coeff .+ sub.soil_water_diffusion_coeff
    jr1 = sub.water_use_efficiency .* sub.plant_uptake_rate
    println("-" ^ 62)
    for (nm, v) in ("d1 + d2" => d12, "j * r1" => jr1)
        @printf("%-30s %10.4g %8.3f %12.4g\n", nm, Statistics.median(v), cv(v),
                maximum(v) / minimum(v))
    end
    np = npinned.(eachrow(sub))
    @printf("pinned at the 1e-4 floor: %d/%d models (%.0f%%)\n",
            count(>(0), np), n, 100 * count(>(0), np) / n)
    rs = ratio.(eachrow(sub))
    @printf("dt/dt_max: median %.2f, range %.2f-%.2f | over 1: %d/%d\n",
            Statistics.median(rs), extrema(rs)..., count(>(1), rs), n)
    return nothing
end

comp = filter(r -> r.hybrid_loss < LOSS_CUT, df)
conv = filter(r -> r.hybrid_loss < LOSS_CUT && r.hybrid_gnorm < GRAD_CUT, df)
banner("Spread")
spread(df; label = "all models (unfiltered — reported for contrast only)")
spread(comp; label = "loss < $LOSS_CUT")
spread(conv; label = "loss < $LOSS_CUT and ‖∇L‖ < $GRAD_CUT")

# ---------------------------------------------------------------------------
if COMPARE !== nothing
    banner("Comparison")
    @printf("%-26s %5s %5s %7s %8s %8s %8s\n",
            "ensemble", "n", "comp", "pinned", "CV l3", "CV d3", "CV d1+d2")
    for (lbl, dir) in ("this run" => RUNDIR, "comparison" => COMPARE)
        d = load(dir)
        c = filter(r -> r.hybrid_loss < LOSS_CUT, d)
        np = npinned.(eachrow(c))
        @printf("%-26s %5d %5d %6.0f%% %8.3f %8.3f %8.3f\n",
                lbl * " (" * basename(dir) * ")", DataFrames.nrow(d), DataFrames.nrow(c),
                100 * count(>(0), np) / DataFrames.nrow(c),
                cv(c.mortality_rate), cv(c.biomass_diffusion_coeff),
                cv(c.surface_water_diffusion_coeff .+ c.soil_water_diffusion_coeff))
    end
end

# ---------------------------------------------------------------------------
# The published Adam ensemble, restricted the same way the accuracy audit does:
# fits whose diffusion coefficients never left the initial O(1) range never
# converged and are not comparable.
banner("Published Adam ensemble")
ptab = read_parameter_table(joinpath(PY_RESULTS, "parameter_history_analysis",
                                     "four_site_final_parameter_values.csv"))
wellres = filter(r -> maximum(paramvector(params_from_row(r))[1:3]) > 10, eachrow(ptab))
@printf("%d of %d published fits are well-resolved\n", length(wellres), DataFrames.nrow(ptab))
@printf("%-30s %10s %8s %12s\n", "parameter", "median", "CV", "max/min")
adam = Dict(n => [row[n] for row in wellres] for n in PARAM_NAMES)
for nm in sort(collect(PARAM_NAMES), by = n -> cv(adam[n]))
    v = adam[nm]
    @printf("%-30s %10.4g %8.3f %12.4g\n", nm, Statistics.median(v), cv(v),
            maximum(v) / minimum(v))
end

#  The published CVs above are only comparable to this run's if both ensembles
#  fit the data equally well, so evaluate the published parameters on the same
#  problem rather than taking the agreement statistics at face value. A tight
#  parameter spread at a *worse* loss means something different from a tight
#  spread at the same loss.
if !("--skip-loss" in ARGS)
    banner("Published fits evaluated on this problem")
    sites = load_sites(DATA_DIR, ["b", "i", "c", "e"]; multiplier = BMAX, T = Float64)
    prob = InverseProblem(sites, CFG; threaded = true)
    ls, gs = Float64[], Float64[]
    for row in wellres
        θ = paramvector(params_from_row(row))
        l, g = loss_and_gradient(prob, θ)
        push!(ls, l)
        push!(gs, sqrt(sum(abs2, g)))
    end
    ok = filter(isfinite, ls)
    @printf("loss   : %.2f - %.2f (median %.2f)   [%d of %d finite]\n",
            extrema(ok)..., Statistics.median(ok), length(ok), length(ls))
    gok = filter(isfinite, gs)
    @printf("‖∇L‖   : %.3g - %.3g (median %.3g)\n",
            extrema(gok)..., Statistics.median(gok))
    conv_ok = filter(r -> isfinite(r.hybrid_loss), conv)
    @printf("\nthis run (loss+grad filtered, n=%d):\n", DataFrames.nrow(conv_ok))
    @printf("loss   : %.2f - %.2f (median %.2f)\n",
            extrema(conv_ok.hybrid_loss)..., Statistics.median(conv_ok.hybrid_loss))
    @printf("‖∇L‖   : %.3g - %.3g (median %.3g)\n",
            extrema(conv_ok.hybrid_gnorm)..., Statistics.median(conv_ok.hybrid_gnorm))
end

println()
