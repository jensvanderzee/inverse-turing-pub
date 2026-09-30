#!/usr/bin/env julia
"""
Bifurcation diagram: equilibrium vegetation density versus annual rainfall.

Port of `bifurcation_parallel.py`. Each surviving model is simulated to
equilibrium at every rainfall level; the best model (lowest held-out MSE) is drawn
in bold with spatial snapshots inset, the rest as a grey ensemble showing how much
the tipping point depends on which fit you believe.

Usage
-----
    julia --project=julia -t auto julia/scripts/bifurcation.jl

Options
    --params <csv>        parameter table
    --metrics <csv>       held-out metrics used to pick the best model
    --history <dir>       parameter histories for the tier-1 filter (.pkl or .json)
    --precip-min 255      rainfall sweep, mm
    --precip-max 355
    --precip-step 3
    --years 1000          simulated years per rainfall level
    --steps-per-week 4
    --init-site a         subsite whose first year provides the initial field
    --out <dir>

Cost: `n_models x n_levels` independent rollouts of `years x 52 x steps_per_week`
Euler steps. The published settings are ~35 models x 34 levels x 208k steps; with
`-t auto` on 32 cores expect a few hours. Use `--years 200` for a quick look —
the branch positions move slightly but the qualitative shape is already there.
"""

using InverseTuring
using Printf
using Plots
import CSV, DataFrames, Statistics

include(joinpath(@__DIR__, "common.jl"))

const PARAM_CSV = string(argval("params", joinpath(PY_RESULTS, "parameter_history_analysis",
                                                   "four_site_final_parameter_values.csv")))
const METRICS_CSV = string(argval("metrics", joinpath(PY_RESULTS, "real_data", "test_results",
                                                      "test_metrics.csv")))
const HISTORY_DIR = string(argval("history", joinpath(PY_RESULTS, "real_data", "models", "parameters")))
const PRECIP_MIN = argfloat("precip-min", 255.0)
const PRECIP_MAX = argfloat("precip-max", 355.0)
const PRECIP_STEP = argfloat("precip-step", 3.0)
const YEARS = argint("years", 1000)
const STEPS_PER_WEEK = argint("steps-per-week", 4)
const INIT_SITE = string(argval("init-site", "a"))
const MULTIPLIER = argfloat("multiplier", 1500.0)
const OUTDIR = string(argval("out", joinpath(OUT_ROOT, "bifurcation")))
const INSET_AT = [265.0, 280.0, 295.0, 310.0, 325.0, 340.0]

banner("Bifurcation sweep")
report_threads()

param_df = read_parameter_table(PARAM_CSV)
model_ids = collect(param_df.model_id)
println("models in table  : ", length(model_ids))

# ---------------------------------------------------------------------------
# Tier-1 filter: drop runs that failed structurally rather than fitting badly.
if isdir(HISTORY_DIR)
    histories = load_parameter_histories(HISTORY_DIR)
    if !isempty(histories)
        kept, dropped = tier1_filter(filter(p -> p.first in model_ids, histories))
        println("\nTier-1 filter (history length + degenerate parameters):")
        for (id, reason) in dropped
            println("  dropped model $id: $reason")
        end
        @printf("  kept %d of %d\n", length(kept), length(histories))
        param_df = param_df[in.(param_df.model_id, Ref(Set(kept))), :]
        model_ids = collect(param_df.model_id)
    end
else
    @warn "no parameter-history directory; skipping the tier-1 filter" HISTORY_DIR
end
isempty(model_ids) && error("no models left after filtering")

# ---------------------------------------------------------------------------
best_id = if isfile(METRICS_CSV)
    m = CSV.read(METRICS_CSV, DataFrames.DataFrame)
    per = DataFrames.combine(DataFrames.groupby(m, :model_id), :mse => Statistics.mean => :mean_mse)
    per = per[in.(per.model_id, Ref(Set(model_ids))), :]
    id = per[argmin(per.mean_mse), :model_id]
    @printf("\nBest model by held-out MSE: %d (%.2f)\n", id, minimum(per.mean_mse))
    id
else
    @warn "no held-out metrics; highlighting the first model instead" METRICS_CSV
    first(model_ids)
end

# Initial condition: a real observed biomass field, so the sweep starts from a
# state the landscape actually occupies rather than from noise.
init_series = load_site(DATA_DIR, INIT_SITE; multiplier = MULTIPLIER, T = Float64)
initial = init_series.observations[1].biomass
@printf("Initial field: %s year %d, %dx%d, mean %.2f\n", init_series.name,
        init_series.observations[1].year, size(initial)..., Statistics.mean(initial))

precip_values = collect(PRECIP_MIN:PRECIP_STEP:PRECIP_MAX)
cfg = SimConfig(steps_per_week = STEPS_PER_WEEK, year_time_units = 1.0)
@printf("Sweep: %d levels (%.0f-%.0f mm) x %d models x %d years\n",
        length(precip_values), PRECIP_MIN, PRECIP_MAX, length(model_ids), YEARS)
@printf("       %d Euler steps per rollout, %d rollouts total\n",
        YEARS * 52 * STEPS_PER_WEEK, length(precip_values) * length(model_ids))

# ---------------------------------------------------------------------------
banner("Simulating")
curves = zeros(Float64, length(model_ids), length(precip_values))
snapshots = Dict{Float64,Matrix{Float64}}()
t0 = time()
for (i, row) in enumerate(eachrow(param_df))
    p = params_from_row(row)
    want = row.model_id == best_id ? INSET_AT : Float64[]
    means, snaps = bifurcation_sweep(p, initial, precip_values;
                                     years = YEARS, cfg = cfg, snapshot_at = want)
    curves[i, :] = means
    merge!(snapshots, snaps)
    @printf("  model %3d done (%d/%d)  %.1f min elapsed\n",
            row.model_id, i, length(model_ids), (time() - t0) / 60)
    flush(stdout)
end
@printf("Sweep finished in %.1f min\n", (time() - t0) / 60)

mkpath(OUTDIR)
out = DataFrames.DataFrame(curves, Symbol.(string.(precip_values)))
DataFrames.insertcols!(out, 1, :model_id => model_ids)
CSV.write(joinpath(OUTDIR, "bifurcation_data.csv"), out)
println("data -> ", joinpath(OUTDIR, "bifurcation_data.csv"))

# ---------------------------------------------------------------------------
banner("Plotting")
best_idx = findfirst(==(best_id), model_ids)

plt = plot(size = (1100, 620), dpi = 200,
           xlabel = "Average annual precipitation [mm]",
           ylabel = "Average vegetation density [g/m²]",
           xlims = (PRECIP_MIN, PRECIP_MAX), legend = :topleft)
for i in eachindex(model_ids)
    i == best_idx && continue
    plot!(plt, precip_values, curves[i, :], color = :silver, lw = 0.8, alpha = 0.6,
          label = i == (best_idx == 1 ? 2 : 1) ? "Other models" : "")
end
plot!(plt, precip_values, curves[best_idx, :], color = :black, lw = 2.5, label = "Best model")
ylims!(plt, 0, Inf)

for pv in sort(collect(keys(snapshots)))
    j = findfirst(v -> isapprox(v, pv; atol = 1e-9), precip_values)
    j === nothing && continue
    scatter!(plt, [pv], [curves[best_idx, j]], mc = :red, ms = 5, label = "")
    vline!(plt, [pv], lc = :red, lw = 0.5, alpha = 0.4, label = "")
end
savefig(plt, joinpath(OUTDIR, "bifurcation_diagram.png"))
savefig(plt, joinpath(OUTDIR, "bifurcation_diagram.pdf"))
println("  bifurcation_diagram.png / .pdf")

# Spatial snapshots as a separate panel. Plots.jl has no clean equivalent of
# matplotlib's AnnotationBbox insets, so the fields are shown side by side
# beneath the curve rather than floating on it.
if !isempty(snapshots)
    keys_sorted = sort(collect(keys(snapshots)))
    vmax = MULTIPLIER * 0.35
    sp = plot(layout = (1, length(keys_sorted)), size = (260 * length(keys_sorted), 300), dpi = 200)
    for (k, pv) in enumerate(keys_sorted)
        f = snapshots[pv]
        heatmap!(sp[k], f, clims = (0, vmax), c = :YlGn, yflip = true, axis = false,
                 colorbar = k == length(keys_sorted),
                 title = @sprintf("%.0f mm\nmean %.1f", pv, Statistics.mean(f)),
                 titlefontsize = 9)
    end
    savefig(sp, joinpath(OUTDIR, "spatial_snapshots.png"))
    savefig(sp, joinpath(OUTDIR, "spatial_snapshots.pdf"))
    println("  spatial_snapshots.png / .pdf")
end

# Where does the ensemble put the collapse? Report the rainfall at which each
# model's biomass first exceeds 5% of its own maximum.
thresholds = Float64[]
for i in eachindex(model_ids)
    row = curves[i, :]
    mx = maximum(row)
    mx <= 0 && continue
    j = findfirst(>(0.05 * mx), row)
    j === nothing || push!(thresholds, precip_values[j])
end
if !isempty(thresholds)
    @printf("\nCollapse threshold across %d models: %.0f-%.0f mm (median %.0f mm)\n",
            length(thresholds), minimum(thresholds), maximum(thresholds),
            Statistics.median(thresholds))
end

println("\nDone. Outputs in ", OUTDIR)
