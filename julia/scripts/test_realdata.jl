#!/usr/bin/env julia
"""
Score fitted models on held-out sites.

Port of `realdata_test_invPDE.py`. Every model in the parameter table is rolled
forward on sites it was never fitted to; the model with the lowest mean delta-MSE
is the one the bifurcation analysis then uses.

Usage
-----
    julia --project=julia -t auto julia/scripts/test_realdata.jl

Options
    --params <csv>        parameter table (default: the published four-site CSV)
    --sites f,k,j         held-out subsites
    --steps-per-week 3    must match the value used at fit time
    --multiplier 1500     NDVI -> biomass scaling
    --out <dir>           output directory
    --plot-best           also render observed/predicted/difference maps

Writes `test_metrics.csv` and `test_summary.json` in the same schema as the
Python version, so downstream scripts accept either.
"""

using InverseTuring
using Printf
using Plots
import CSV, DataFrames, Statistics, Random

include(joinpath(@__DIR__, "common.jl"))

const PARAM_CSV = string(argval("params", joinpath(PY_RESULTS, "parameter_history_analysis",
                                                   "four_site_final_parameter_values.csv")))
const TEST_SITES = arglist("sites", ["f", "k", "j"])
const STEPS_PER_WEEK = argint("steps-per-week", 3)
const MULTIPLIER = argfloat("multiplier", 1500.0)
const OUTDIR = string(argval("out", joinpath(OUT_ROOT, "test_results")))
const PLOT_BEST = argflag("plot-best")

banner("Evaluating fitted models on held-out sites")
report_threads()
println("parameters       : ", PARAM_CSV)
println("held-out sites   : ", join(TEST_SITES, ", "))
println("steps per week   : ", STEPS_PER_WEEK)

isfile(PARAM_CSV) || error("parameter table not found: $PARAM_CSV")
param_df = read_parameter_table(PARAM_CSV)
println("models           : ", DataFrames.nrow(param_df))

# Float32 matches the precision the reference results were produced at; the
# metrics are insensitive to this choice at the digits reported.
sites = load_sites(DATA_DIR, TEST_SITES; multiplier = MULTIPLIER, T = Float32)
cfg = SimConfig(steps_per_week = STEPS_PER_WEEK, year_time_units = 1.0)
for s in sites
    ys = years(s)
    @printf("  %-12s %2d years %d-%d  %dx%d\n", s.name, length(s),
            minimum(ys), maximum(ys), size(s)...)
end

# ---------------------------------------------------------------------------
banner("Rolling models forward")
rows = param_df[:, :]
ids = rows.model_id
models = [params_from_row(r) for r in eachrow(rows)]

# One (model, site) pair per task; independent, so thread across all of them.
tasks = [(i, j) for i in eachindex(models), j in eachindex(sites)] |> vec
records = Vector{NamedTuple}(undef, length(tasks))
done = Threads.Atomic{Int}(0)

t0 = time()
Threads.@threads for k in eachindex(tasks)
    i, j = tasks[k]
    m = evaluate(models[i], sites[j], cfg)
    records[k] = (model_id = ids[i], site = sites[j].name, mse = m.mse, mae = m.mae,
                  correlation = m.correlation, num_transitions = m.num_transitions,
                  turing_value = turing_value(models[i]))
    n = Threads.atomic_add!(done, 1) + 1
    n % 25 == 0 && @printf("  %d/%d evaluations\n", n, length(tasks))
end
@printf("Completed %d evaluations in %.1f s\n", length(tasks), time() - t0)

results_df = DataFrames.DataFrame(records)
mkpath(OUTDIR)
csv_path = joinpath(OUTDIR, "test_metrics.csv")
CSV.write(csv_path, results_df)
println("metrics -> ", csv_path)

# ---------------------------------------------------------------------------
banner("Summary")
# A parameter set can drive the explicit integrator unstable, giving a non-finite
# score. Those runs are counted and excluded rather than allowed to poison the
# summary statistics — they are a result about the model, not a measurement of it.
finite_stats(v) = (f = filter(isfinite, v); (mean = isempty(f) ? NaN : Statistics.mean(f),
                                             std = length(f) < 2 ? NaN : Statistics.std(f),
                                             min = isempty(f) ? NaN : minimum(f),
                                             max = isempty(f) ? NaN : maximum(f),
                                             n = length(f), n_bad = length(v) - length(f)))

summary = Dict{String,Any}()
for site in sort(unique(results_df.site))
    sub = results_df[results_df.site .== site, :]
    m, a, c = finite_stats(sub.mse), finite_stats(sub.mae), finite_stats(sub.correlation)
    stats = Dict(
        "mse_mean" => m.mean, "mse_std" => m.std, "mse_min" => m.min, "mse_max" => m.max,
        "mae_mean" => a.mean, "mae_std" => a.std,
        "corr_mean" => c.mean, "corr_std" => c.std,
        "num_models" => DataFrames.nrow(sub), "num_diverged" => m.n_bad)
    summary[site] = stats
    @printf("\n%s:\n", site)
    @printf("  MSE  %10.2f +/- %8.2f   (min %.2f, max %.2f)\n",
            m.mean, m.std, m.min, m.max)
    @printf("  MAE  %10.2f +/- %8.2f\n", a.mean, a.std)
    @printf("  Corr %10.4f +/- %8.4f\n", c.mean, c.std)
    m.n_bad > 0 && @printf("  %d of %d models diverged (non-finite score) and are excluded\n",
                           m.n_bad, DataFrames.nrow(sub))
end
summary["overall"] = Dict("mse_mean" => finite_stats(results_df.mse).mean,
                          "mae_mean" => finite_stats(results_df.mae).mean,
                          "corr_mean" => finite_stats(results_df.correlation).mean,
                          "num_diverged" => finite_stats(results_df.mse).n_bad)
save_json(joinpath(OUTDIR, "test_summary.json"), summary)

# Best model = lowest mean MSE across held-out sites. A model that diverged on any
# site cannot be best, so non-finite scores propagate through the mean and lose.
per_model = DataFrames.combine(DataFrames.groupby(results_df, :model_id),
                               :mse => Statistics.mean => :mean_mse,
                               :correlation => Statistics.mean => :mean_corr)
usable = per_model[isfinite.(per_model.mean_mse), :]
DataFrames.nrow(usable) == 0 && error("every model diverged on at least one held-out site")
DataFrames.nrow(usable) < DataFrames.nrow(per_model) &&
    @printf("\n%d of %d models diverged on at least one site and cannot be ranked\n",
            DataFrames.nrow(per_model) - DataFrames.nrow(usable), DataFrames.nrow(per_model))
best_row = usable[argmin(usable.mean_mse), :]
best_id = best_row.model_id
@printf("\nBest model: %d  (mean held-out MSE %.2f, mean correlation %.4f)\n",
        best_id, best_row.mean_mse, best_row.mean_corr)
best_params = models[findfirst(==(best_id), ids)]
show(stdout, MIME"text/plain"(), best_params)
save_json(joinpath(OUTDIR, "best_model.json"),
          Dict("model_id" => best_id, "mean_mse" => best_row.mean_mse,
               "params" => paramdict(best_params),
               "turing_value" => turing_value(best_params)))

# ---------------------------------------------------------------------------
banner("Plotting")
plt = plot(layout = (1, 3), size = (1300, 450), dpi = 150)
sitenames = sort(unique(results_df.site))
rng = Random.Xoshiro(0)
for (idx, (col, label)) in enumerate(zip((:mse, :mae, :correlation), ("MSE", "MAE", "Correlation")))
    logscale = col !== :correlation
    for (x, s) in enumerate(sitenames)
        v = filter(isfinite, results_df[results_df.site .== s, col])
        isempty(v) && continue
        # Horizontal jitter so overlapping models stay visible.
        scatter!(plt[idx], x .+ 0.12 .* randn(rng, length(v)), v,
                 ms = 3, alpha = 0.5, mc = :steelblue, legend = false)
        m, sd = Statistics.mean(v), Statistics.std(v)
        scatter!(plt[idx], [x], [m], yerror = [sd], ms = 7, mc = :black, legend = false)
    end
    plot!(plt[idx], title = label, xlabel = "held-out site", ylabel = label,
          xticks = (1:length(sitenames), sitenames), xlims = (0.5, length(sitenames) + 0.5),
          yscale = logscale ? :log10 : :identity)
end
savefig(plt, joinpath(OUTDIR, "test_metrics.png"))
println("  test_metrics.png")

if PLOT_BEST
    for s in sites
        n = length(s) - 1
        n == 0 && continue
        state = simstate(s.observations[1].biomass)
        preds = [Float64.(s.observations[1].biomass)]
        for k in 1:n
            simulate_year!(state, best_params, s.observations[k].weekly_precipitation, cfg)
            push!(preds, Float64.(state.biomass))
        end
        vmax = MULTIPLIER * 0.5
        pl = plot(layout = (n + 1, 3), size = (900, 260 * (n + 1)), dpi = 130)
        for i in 0:n
            obs = Float64.(s.observations[i + 1].biomass)
            pred = preds[i + 1]
            d = pred .- obs
            lim = max(maximum(abs, d), 1e-6)
            yr = s.observations[i + 1].year
            heatmap!(pl[3i + 1], obs, clims = (0, vmax), c = :RdYlGn, title = "obs $yr",
                     axis = false, colorbar = false, yflip = true)
            heatmap!(pl[3i + 2], pred, clims = (0, vmax), c = :RdYlGn, title = "pred $yr",
                     axis = false, colorbar = false, yflip = true)
            heatmap!(pl[3i + 3], d, clims = (-lim, lim), c = :RdBu, title = "diff $yr",
                     axis = false, colorbar = false, yflip = true)
        end
        savefig(pl, joinpath(OUTDIR, "best_model_predictions_$(s.name).png"))
        println("  best_model_predictions_$(s.name).png")
    end
end

println("\nDone. Outputs in ", OUTDIR)
