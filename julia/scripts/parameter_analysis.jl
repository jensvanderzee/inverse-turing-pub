#!/usr/bin/env julia
"""
How much do independent fits agree with each other?

Port of `realdata_parameter_analysis.py`. Reads parameter histories from a set of
fits — Python `.pkl` or Julia `.json`, both work — and reports, per coefficient,
the spread across runs, the agreement score, and the trajectory over training.

The quantity to watch is whether the composite growth efficiency is more tightly
constrained than the individual rate constants it is built from — the signature of
a problem where the data identify a *combination* of parameters rather than the
parameters themselves.

Whether it is depends on which runs you pool, and the script reports what it finds
rather than assuming. Over the 47 runs that survive the degenerate-value filter it
is **not** (composite CV 1.63, individual rates 0.64–1.14) — that population still
contains fits that converged to entirely different regimes. Over the 32 runs that
also pass [`tier1_filter`](@ref) it clearly is (composite CV 0.03 against
0.10–0.22). Use `--filter tier1` for the second case, and quote which filter you
used, because the conclusion flips between them.

Usage
-----
    julia --project=julia julia/scripts/parameter_analysis.jl

Options
    --history <dir>   directory of model_XX_params.{pkl,json}
    --sites b,i,c,e   training subsites, used to compute the biomass level B
    --max-models 100  cap on how many runs to include
    --filter analysis `analysis` drops runs with a degenerate final parameter
                      (the rule realdata_parameter_analysis.py uses); `tier1`
                      additionally drops runs that stopped early
    --out <dir>

Defaults reproduce the published `four_site_agreement_table.csv`.
"""

using InverseTuring
using Printf
using Plots
import CSV, DataFrames, Statistics

include(joinpath(@__DIR__, "common.jl"))

# NOTE: this is `results/real_data/parameters`, NOT `results/real_data/models/parameters`.
# Both directories exist and contain identically named files from different training
# campaigns. `realdata_parameter_analysis.py` reads this one (results_dir =
# "./results/real_data"); `bifurcation_parallel.py` reads the `models/` one. They
# disagree — e.g. model 82 is a diverged 3-snapshot run here and a healthy
# 751-snapshot run there. Pass --history explicitly if you mean the other set.
const HISTORY_DIR = string(argval("history", joinpath(PY_RESULTS, "real_data", "parameters")))
const SITES = arglist("sites", ["b", "i", "c", "e"])
const MAX_MODELS = argint("max-models", 100)
const FILTER = Symbol(argval("filter", "analysis"))
FILTER in (:analysis, :tier1) || error("--filter must be analysis or tier1, got $FILTER")
const MULTIPLIER = argfloat("multiplier", 1500.0)
const OUTDIR = string(argval("out", joinpath(OUT_ROOT, "parameter_analysis")))

banner("Parameter agreement analysis")
println("histories        : ", HISTORY_DIR)
isdir(HISTORY_DIR) || error("history directory not found: $HISTORY_DIR")

histories = load_parameter_histories(HISTORY_DIR)
isempty(histories) && error("no model_XX_params.{pkl,json} found in $HISTORY_DIR")
@printf("loaded %d parameter histories (%d-%d snapshots each)\n", length(histories),
        minimum(length, values(histories)), maximum(length, values(histories)))

kept, dropped = FILTER === :tier1 ? tier1_filter(histories) : drop_degenerate(histories)
println("\nFilter (:$FILTER):")
for (id, reason) in dropped
    println("  dropped model $id: $reason")
end
@printf("  kept %d of %d\n", length(kept), length(histories))
isempty(kept) && error("no usable runs")

selected = sort(kept)[1:min(MAX_MODELS, length(kept))]
@printf("using %d runs: %s\n", length(selected), join(selected, ", "))

# ---------------------------------------------------------------------------
banner("Biomass level for the composite parameter")
B = mean_training_biomass(DATA_DIR, SITES; multiplier = MULTIPLIER)
@printf("mean training biomass over subsites %s: B = %.4f\n", join(SITES, ","), B)

# ---------------------------------------------------------------------------
banner("Cross-run agreement")
finals = Dict(id => RietkerkParams(histories[id][end]) for id in selected)
values_by_name = Dict{String,Vector{Float64}}(
    n => [getfield(finals[id], i) for id in selected] for (i, n) in enumerate(PARAM_NAMES))
values_by_name["turing_value"] = [turing_value(finals[id]) for id in selected]
values_by_name["composite_value"] = [composite_value(finals[id], B) for id in selected]

tbl = agreement_table(values_by_name; ground_truth = REALDATA_REFERENCE)
mkpath(OUTDIR)
CSV.write(joinpath(OUTDIR, "agreement_table.csv"), tbl)
show(stdout, MIME"text/plain"(), tbl)
println("\n")

write_parameter_table(joinpath(OUTDIR, "final_parameter_values.csv"), selected,
                      [finals[id] for id in selected];
                      extra = Dict("turing_value" => values_by_name["turing_value"],
                                   "composite_value" => values_by_name["composite_value"]))

# The comparison the analysis exists to make.
comp_row = only(filter(r -> r.parameter == "composite_value", eachrow(tbl)))
rate_rows = filter(r -> r.parameter in ("infiltration_rate", "plant_uptake_rate",
                                        "evaporation_rate", "seepage_rate"), tbl)
@printf("composite value : cv = %.3f  (agreement %.3f)\n", comp_row.cv, comp_row.agreement_score)
@printf("its four rate constants: cv = %.3f - %.3f\n",
        minimum(rate_rows.cv), maximum(rate_rows.cv))
if comp_row.cv < minimum(rate_rows.cv)
    println("-> the composite is better constrained than any of its factors:")
    println("   over this set of runs the data identify the combination, not the rates.")
else
    println("-> the composite is NOT better constrained over this set of runs.")
    println("   With --filter $(FILTER === :tier1 ? "analysis" : "tier1") the pooled runs differ; the")
    println("   comparison is sensitive to which fits are included, so quote the filter.")
end

n_turing = count(<(0), values_by_name["turing_value"])
@printf("\n%d of %d runs predict a Turing instability (negative turing value)\n",
        n_turing, length(selected))

# ---------------------------------------------------------------------------
banner("Plotting")

# Parameter trajectories over training, one panel per coefficient.
plt = plot(layout = (3, 3), size = (1250, 900), dpi = 150)
for (i, name) in enumerate(PARAM_NAMES)
    for id in selected
        h = histories[id]
        epochs = [s["epoch"] for s in h]
        vals = [s[name] for s in h]
        plot!(plt[i], epochs, max.(vals, 1e-5), lw = 1, alpha = 0.7, label = "")
    end
    gt = paramdict(REALDATA_REFERENCE)[name]
    hline!(plt[i], [gt], lc = :black, ls = :dash, lw = 1.5, label = "")
    plot!(plt[i], title = replace(name, "_" => " "), titlefontsize = 9,
          yscale = :log10, xlabel = i > 6 ? "epoch" : "", ylabel = "value", legend = false)
end
savefig(plt, joinpath(OUTDIR, "parameter_histories.png"))

# Agreement summary: coefficient of variation per quantity.
rows = filter(r -> r.parameter != "turing_value", tbl)
plt2 = bar(1:DataFrames.nrow(rows), rows.cv, legend = false, dpi = 150, size = (950, 520),
           xticks = (1:DataFrames.nrow(rows), replace.(rows.parameter, "_" => " ")),
           xrotation = 40, ylabel = "coefficient of variation across runs",
           title = "Cross-run agreement (lower is better)",
           color = [r == "composite_value" ? :seagreen : :steelblue for r in rows.parameter])
savefig(plt2, joinpath(OUTDIR, "agreement_summary.png"))

# Composite value per run, against the spread of a single rate constant.
plt3 = plot(size = (900, 480), dpi = 150, xlabel = "run", ylabel = "value (normalised to mean)",
            title = "Composite vs individual rates, per run")
for (name, color) in (("composite_value", :seagreen), ("infiltration_rate", :orangered),
                      ("plant_uptake_rate", :steelblue), ("water_use_efficiency", :purple))
    v = values_by_name[name]
    plot!(plt3, 1:length(selected), v ./ Statistics.mean(v), lw = 2, marker = :circle,
          ms = 4, color = color, label = replace(name, "_" => " "))
end
savefig(plt3, joinpath(OUTDIR, "composite_vs_rates.png"))

println("  parameter_histories.png, agreement_summary.png, composite_vs_rates.png")
println("  agreement_table.csv, final_parameter_values.csv")
println("\nDone. Outputs in ", OUTDIR)
