#!/usr/bin/env julia
"""
Check the Julia port against artefacts produced by the original PyTorch code.

Nothing here is re-fitted. Every target below was written by a Python script that
is still in the repository, so agreement means the two implementations agree on
the data pipeline, the PDE discretisation, and the evaluation metrics — not merely
that the Julia code runs.

Checks, in increasing order of how much of the pipeline they cover:

1. `data_info.json`   — NDVI -> biomass conversion and site statistics.
2. parameter CSV      — the Turing diagnostic, from the same coefficients.
3. `test_metrics.csv` — a 9-year rollout on three held-out sites, scored with the
                        same delta-MSE / MAE / correlation as `realdata_test_invPDE.py`.
4. `four_site_agreement_table.csv` — reading the Python pickled parameter
                        histories, applying the same run filter, and recomputing
                        every agreement statistic.
5. `bifurcation_data.csv` — 1000-year equilibrium rollouts (208 000 Euler steps
                        each). Slow; pass `--full` for more models and levels.

Element type is `Float32` throughout, because that is what PyTorch used; the
package defaults to `Float64` for fitting.

Usage:
    julia --project=julia julia/scripts/validate_against_python.jl
"""

using InverseTuring
using Printf
import CSV, DataFrames, JSON, Statistics

const REPO = normpath(joinpath(@__DIR__, "..", ".."))
const DATA_DIR = joinpath(REPO, "data")
const PARAM_CSV = joinpath(REPO, "results", "parameter_history_analysis",
                           "four_site_final_parameter_values.csv")
const TEST_METRICS = joinpath(REPO, "results", "real_data", "test_results", "test_metrics.csv")
const DATA_INFO = joinpath(REPO, "results", "real_data", "data_info.json")

const MULTIPLIER = 1500.0
const STEPS_PER_WEEK = 4
const TRAIN_SITES = ["b", "i", "c", "e"]
const TEST_SITES = ["f", "k", "j"]

const npass = Ref(0)
const nfail = Ref(0)

function check(label, got, want; rtol)
    ok = isapprox(got, want; rtol = rtol)
    ok ? npass[] += 1 : nfail[] += 1
    rel = want == 0 ? abs(got) : abs(got - want) / abs(want)
    @printf("  %-46s %18.10g  vs %18.10g   rel %.2e  %s\n",
            label, got, want, rel, ok ? "ok" : "FAIL")
    return ok
end

println("=" ^ 118)
println("Validating InverseTuring.jl against PyTorch reference outputs")
println("=" ^ 118)

# ---------------------------------------------------------------------------
# 1. Data pipeline: NDVI -> biomass
# ---------------------------------------------------------------------------
println("\n[1] NDVI -> biomass conversion and site statistics  (target: results/real_data/data_info.json)")
if isfile(DATA_INFO)
    ref = JSON.parsefile(DATA_INFO)["global_stats"]
    sites = load_sites(DATA_DIR, TRAIN_SITES; multiplier = MULTIPLIER, T = Float32)
    stats = biomass_stats(sites)
    check("min per-image mean biomass", stats.min_biomass, ref["min_biomass"]; rtol = 1e-6)
    check("max per-image mean biomass", stats.max_biomass, ref["max_biomass"]; rtol = 1e-6)
    check("mean per-image mean biomass", stats.mean_biomass, ref["mean_biomass"]; rtol = 1e-6)
    check("std per-image mean biomass", stats.std_biomass, ref["std_biomass"]; rtol = 1e-6)
    check("mean annual precipitation [mm]", stats.mean_precip, ref["mean_precip"]; rtol = 1e-12)
    check("min annual precipitation [mm]", stats.min_precip, ref["min_precip"]; rtol = 1e-12)
else
    println("  SKIPPED - $DATA_INFO not found")
end

# ---------------------------------------------------------------------------
# 2. Turing diagnostic
# ---------------------------------------------------------------------------
println("\n[2] Turing instability diagnostic  (target: four_site_final_parameter_values.csv)")
param_df = read_parameter_table(PARAM_CSV)
for row in eachrow(param_df)
    p = params_from_row(row)
    check("turing_value, model $(row.model_id)", turing_value(p), row.turing_value; rtol = 1e-10)
end

# ---------------------------------------------------------------------------
# 3. Full rollout on held-out sites
# ---------------------------------------------------------------------------
println("\n[3] Nine-year rollout on held-out sites  (target: results/real_data/test_results/test_metrics.csv)")
println("    9 transitions x 52 weeks x $STEPS_PER_WEEK steps = ",
        9 * 52 * STEPS_PER_WEEK, " Euler steps per site, Float32")

if isfile(TEST_METRICS)
    ref_df = CSV.read(TEST_METRICS, DataFrames.DataFrame)
    cfg = SimConfig(steps_per_week = STEPS_PER_WEEK, year_time_units = 1.0)
    test_data = Dict(s.name => s for s in
                     load_sites(DATA_DIR, TEST_SITES; multiplier = MULTIPLIER, T = Float32))

    # Evaluating every model against every site is the full published table; keep
    # it to a representative subset unless RUN_ALL is set, since each entry is a
    # 1872-step rollout.
    model_ids = get(ENV, "RUN_ALL", "0") == "1" ?
                unique(ref_df.model_id) : unique(ref_df.model_id)[1:min(3, end)]

    for mid in model_ids
        prow = only(filter(r -> r.model_id == mid, eachrow(param_df)))
        p = params_from_row(prow)
        println("  model $mid:")
        for site_name in sort(collect(keys(test_data)))
            got = evaluate(p, test_data[site_name], cfg)
            refrow = only(filter(r -> r.model_id == mid && r.site == site_name, eachrow(ref_df)))
            check("    $site_name  delta-MSE", got.mse, refrow.mse; rtol = 1e-4)
            check("    $site_name  delta-MAE", got.mae, refrow.mae; rtol = 1e-4)
            check("    $site_name  correlation", got.correlation, refrow.correlation; rtol = 1e-4)
        end
    end
else
    println("  SKIPPED - $TEST_METRICS not found")
end

# ---------------------------------------------------------------------------
# 4. Cross-run agreement statistics, from the Python pickled histories
# ---------------------------------------------------------------------------
println("\n[4] Cross-run agreement  (target: four_site_agreement_table.csv)")
const AGREE_CSV = joinpath(REPO, "results", "parameter_history_analysis",
                           "four_site_agreement_table.csv")
# realdata_parameter_analysis.py reads results_dir/"parameters" with
# results_dir = "results/real_data" -- not the models/ subdirectory that
# bifurcation_parallel.py uses. The two hold different runs under the same names.
const HIST_DIR = joinpath(REPO, "results", "real_data", "parameters")

if isfile(AGREE_CSV) && isdir(HIST_DIR)
    ref = CSV.read(AGREE_CSV, DataFrames.DataFrame)
    histories = load_parameter_histories(HIST_DIR)
    kept, _ = drop_degenerate(histories)      # the rule that script applies
    println("    kept $(length(kept)) of $(length(histories)) runs")

    B = mean_training_biomass(DATA_DIR, TRAIN_SITES; multiplier = MULTIPLIER)
    check("mean training biomass B", B, 150.6342; rtol = 1e-6)

    finals = Dict(id => RietkerkParams(histories[id][end]) for id in kept)
    vals = Dict{String,Vector{Float64}}(
        n => [getfield(finals[id], i) for id in kept] for (i, n) in enumerate(PARAM_NAMES))
    vals["turing_value"] = [turing_value(finals[id]) for id in kept]
    vals["composite_value"] = [composite_value(finals[id], B) for id in kept]
    tbl = agreement_table(vals; ground_truth = REALDATA_REFERENCE)

    # The Python CSV labels rows in Title Case; map back to the canonical names.
    key(s) = replace(lowercase(strip(s)), " " => "_")
    refmap = Dict(key(r[1]) => r for r in eachrow(ref))
    for row in eachrow(tbl)
        haskey(refmap, row.parameter) || continue
        r = refmap[row.parameter]
        # The published table is rounded to 4 decimals, so compare at that level.
        check("$(row.parameter) mean", round(row.mean; digits = 4), r["Mean"]; rtol = 1e-4)
        check("$(row.parameter) cv", round(row.cv; digits = 4), r["CV"]; rtol = 1e-4)
    end
else
    println("  SKIPPED - agreement table or history directory not found")
end

# ---------------------------------------------------------------------------
# 5. Long equilibrium rollouts
# ---------------------------------------------------------------------------
println("\n[5] 1000-year equilibrium rollouts  (target: bifurcation_data.csv)")
const BIF_CSV = joinpath(REPO, "results", "real_data", "bifurcation", "bifurcation_data.csv")
const FULL = "--full" in ARGS

if isfile(BIF_CSV) && isfile(PARAM_CSV)
    py = CSV.read(BIF_CSV, DataFrames.DataFrame)
    # 285 mm sits on the steep part of the branch, 330 mm well above it; the
    # collapsed branch below ~275 mm is a decay through nine orders of magnitude
    # and is only included in --full.
    probe_models = FULL ? py.model_id[1:min(5, end)] : py.model_id[1:1]
    probe_levels = FULL ? [270.0, 285.0, 300.0, 330.0] : [285.0, 330.0]
    println("    ", length(probe_models), " model(s) x ", length(probe_levels),
            " level(s), 208 000 Euler steps each, Float32",
            FULL ? "" : "  (pass --full for more)")

    init = load_site(DATA_DIR, "a"; multiplier = MULTIPLIER, T = Float32).observations[1].biomass
    bcfg = SimConfig(steps_per_week = 4, year_time_units = 1.0)

    for mid in probe_models
        prow = only(filter(r -> r.model_id == mid, eachrow(param_df)))
        pyrow = only(filter(r -> r.model_id == mid, eachrow(py)))
        means, _ = bifurcation_sweep(params_from_row(prow), init, probe_levels;
                                     years = 1000, cfg = bcfg)
        for (j, lv) in enumerate(probe_levels)
            want = Float64(pyrow[Symbol(string(lv))])
            # Below the tipping point the state has decayed to ~1e-9 or less, so
            # a looser tolerance there reflects float32 dynamic range, not a
            # disagreement about the dynamics.
            tol = want < 1e-6 ? 1e-3 : 1e-4
            check("model $mid @ $(Int(lv)) mm", means[j], want; rtol = tol)
        end
    end
else
    println("  SKIPPED - $BIF_CSV not found")
end

println("\n" * "=" ^ 118)
@printf("%d checks passed, %d failed\n", npass[], nfail[])
println("=" ^ 118)
exit(nfail[] == 0 ? 0 : 1)
