#!/usr/bin/env julia
"""
Extrapolation test: does a model fitted on 10 years of one rainfall regime still
behave sensibly 1000 years out, and under rainfall it never saw?

Port of `invPDE_1site_extrapolation.py`. The point of fitting a mechanistic PDE
rather than a black-box network is that it should stay physical outside the
training envelope; this script is where that claim is checked.

Usage
-----
    julia --project=julia -t auto julia/scripts/extrapolation.jl --runs <dir>

Options
    --runs <dir>          directory of run_*.json from train_synthetic.jl
    --params <csv>        alternative: a parameter CSV, with --model-id
    --model-id <int>      which row of the CSV to use
    --years 1000          rollout length
    --grid 128
    --steps-per-week 1
    --preset 1site|4site  sets the true parameters and the year length
    --out <dir>
"""

using InverseTuring
using Printf
using Plots
using Random
import Statistics, JSON

include(joinpath(@__DIR__, "common.jl"))

const PRESET = string(argval("preset", "1site"))
const RUNS_DIR = argval("runs", nothing)
const PARAM_CSV = argval("params", nothing)
const MODEL_ID = argval("model-id", nothing)
const YEARS = argint("years", 1000)
const GRID = argint("grid", 128)
const STEPS_PER_WEEK = argint("steps-per-week", 1)
const OUTDIR = string(argval("out", joinpath(OUT_ROOT, "extrapolation_$PRESET")))

const EXPERIMENT = PRESET == "4site" ?
    (truth = SYNTHETIC_TRUTH, equilibrium_precip = 21.0 * 0.75, year_time_units = 1.5,
     regimes = [("below range", 15.0 * 0.75), ("in range", 21.0 * 0.75), ("above range", 27.0 * 0.75)]) :
    (truth = SYNTHETIC_TRUTH_1SITE, equilibrium_precip = 19.0, year_time_units = 1.0,
     regimes = [("below range", 14.5), ("in range", 20.5), ("above range", 25.5)])

banner("Extrapolation test ($PRESET)")
report_threads()

# ---------------------------------------------------------------------------
# Pick the fitted model to probe: best run in a results directory, or a CSV row.
fitted, label = if RUNS_DIR !== nothing
    dir = string(RUNS_DIR)
    isdir(dir) || error("runs directory not found: $dir")
    files = filter(f -> endswith(f, ".json"), readdir(dir; join = true))
    isempty(files) && error("no run_*.json under $dir")
    runs = [(f, JSON.parsefile(f)) for f in files]
    usable = filter(r -> haskey(r[2], "final_loss") && r[2]["final_loss"] isa Real &&
                         isfinite(r[2]["final_loss"]), runs)
    isempty(usable) && error("no runs with a finite final loss in $dir")
    best = argmin(r -> r[2]["final_loss"], usable)
    @printf("Best of %d runs: %s (final loss %.6g, %d skipped as non-finite)\n",
            length(runs), basename(best[1]), best[2]["final_loss"], length(runs) - length(usable))
    (RietkerkParams(best[2]["final_params"]), basename(best[1]))
elseif PARAM_CSV !== nothing
    df = read_parameter_table(string(PARAM_CSV))
    row = MODEL_ID === nothing ? df[1, :] :
          only(filter(r -> r.model_id == parse(Int, string(MODEL_ID)), eachrow(df)))
    (params_from_row(row), "model $(row.model_id)")
else
    error("give either --runs <dir> or --params <csv>")
end

println("\nFitted parameters ($label):")
show(stdout, MIME"text/plain"(), fitted)
@printf("turing value %.6f   %s\n", turing_value(fitted),
        turing_value(fitted) < 0 ? "(patterning possible)" : "(no Turing instability)")
println("\nTrue parameters:")
show(stdout, MIME"text/plain"(), EXPERIMENT.truth)

# ---------------------------------------------------------------------------
banner("Building initial state")
cfg = SimConfig(steps_per_week = STEPS_PER_WEEK, year_time_units = EXPERIMENT.year_time_units)
init = equilibrium_biomass(EXPERIMENT.truth, (GRID, GRID);
                           precipitation = EXPERIMENT.equilibrium_precip,
                           years = 100, cfg = SimConfig(steps_per_week = 3,
                                                        year_time_units = EXPERIMENT.year_time_units),
                           rng = Xoshiro(42), T = Float64)
@printf("Equilibrium biomass: mean %.4f, range %.4f - %.4f\n",
        Statistics.mean(init), minimum(init), maximum(init))

# ---------------------------------------------------------------------------
banner("Rolling out $YEARS years per regime")

# Both the fitted and the true model are run, so the plot shows not just whether
# the fit stays bounded but whether it agrees with the system it was fitted to.
trajectories = Dict{String,Dict{String,Vector{Float64}}}()
fields = Dict{String,Dict{String,Matrix{Float64}}}()

for (name, annual) in EXPERIMENT.regimes
    weekly = sinusoidal_weekly_precip(annual; peak_week = 26.0, amplitude_fraction = 0.7)
    trajectories[name] = Dict{String,Vector{Float64}}()
    fields[name] = Dict{String,Matrix{Float64}}()
    for (which, p) in (("fitted", fitted), ("truth", EXPERIMENT.truth))
        state = simstate(init)
        means = Float64[Statistics.mean(state.biomass)]
        simulate_years!(state, p, weekly, cfg, YEARS;
                        callback = (_, s) -> push!(means, Statistics.mean(s.biomass)))
        trajectories[name][which] = means
        fields[name][which] = copy(state.biomass)
    end
    f = trajectories[name]["fitted"]
    t = trajectories[name]["truth"]
    @printf("  %-12s annual %5.1f mm  final biomass: fitted %10.4f   truth %10.4f\n",
            name, annual, f[end], t[end])
    if !isfinite(f[end])
        @printf("      fitted model blew up (non-finite) - it does not extrapolate\n")
    elseif t[end] > 1e-6
        @printf("      ratio fitted/truth = %.3f\n", f[end] / t[end])
    end
end

# ---------------------------------------------------------------------------
banner("Plotting")
mkpath(OUTDIR)
n = length(EXPERIMENT.regimes)
plt = plot(layout = (1, n), size = (420 * n, 400), dpi = 150)
for (k, (name, annual)) in enumerate(EXPERIMENT.regimes)
    yrs = 0:YEARS
    plot!(plt[k], yrs, trajectories[name]["fitted"], lw = 2, color = :steelblue, label = "fitted")
    plot!(plt[k], yrs, trajectories[name]["truth"], lw = 2, color = :black,
          ls = :dash, label = "ground truth")
    vline!(plt[k], [10], lc = :gray, ls = :dot, label = "training horizon")
    plot!(plt[k], title = @sprintf("%s\nannual = %.1f mm", name, annual),
          xlabel = "year", ylabel = k == 1 ? "mean biomass" : "",
          titlefontsize = 10, legend = k == 1 ? :best : false)
end
savefig(plt, joinpath(OUTDIR, "extrapolation_trajectories.png"))
savefig(plt, joinpath(OUTDIR, "extrapolation_trajectories.pdf"))

vmax = maximum(maximum(f) for d in values(fields) for f in values(d) if all(isfinite, f); init = 1.0)
sp = plot(layout = (2, n), size = (330 * n, 640), dpi = 150)
for (k, (name, annual)) in enumerate(EXPERIMENT.regimes)
    for (r, which) in enumerate(("fitted", "truth"))
        f = fields[name][which]
        idx = (r - 1) * n + k
        heatmap!(sp[idx], all(isfinite, f) ? f : zeros(size(f)), clims = (0, vmax),
                 c = :YlGn, yflip = true, axis = false, colorbar = (k == n),
                 title = @sprintf("%s / %s\nmean %.3f", name, which, Statistics.mean(f)),
                 titlefontsize = 9)
    end
end
savefig(sp, joinpath(OUTDIR, "extrapolation_final_fields.png"))

save_json(joinpath(OUTDIR, "extrapolation_results.json"), Dict(
    "source" => label, "years" => YEARS, "preset" => PRESET,
    "fitted_params" => paramdict(fitted), "true_params" => paramdict(EXPERIMENT.truth),
    "trajectories" => trajectories,
    "final_means" => Dict(name => Dict(w => Statistics.mean(fields[name][w]) for w in ("fitted", "truth"))
                          for (name, _) in EXPERIMENT.regimes)))

println("  extrapolation_trajectories.png / .pdf")
println("  extrapolation_final_fields.png")
println("\nDone. Outputs in ", OUTDIR)
