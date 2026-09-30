#!/usr/bin/env julia
"""
Fit the PDE parameters to real satellite data.

Port of `realdata_train_invPDE.py`. Trains several models from random restarts on
the training subsites, writes one JSON per run plus a pooled parameter CSV, and
saves a loss-curve figure.

Usage
-----
    julia --project=julia -t auto julia/scripts/train_realdata.jl

Options (defaults match the Python entry point)
    --sites b,i,c,e       training subsites
    --models 10           number of random restarts
    --epochs 7500         Adam epochs per restart
    --steps-per-week 3    Euler sub-steps per forcing week
    --lr 0.2              initial learning rate
    --multiplier 1500     NDVI -> biomass scaling
    --out <dir>           output directory
    --parallel runs       :runs (thread over restarts) | :sites | :none

Timing: with `-t auto` on a 32-core machine expect roughly 1-2 h per restart at
7500 epochs and 3 steps/week. Start with `--models 2 --epochs 200` to check the
pipeline before committing to a full run.
"""

using InverseTuring
using Printf
using Plots
import Statistics

include(joinpath(@__DIR__, "common.jl"))

const SITES = arglist("sites", ["b", "i", "c", "e"])
const NMODELS = argint("models", 10)
const EPOCHS = argint("epochs", 7500)
const STEPS_PER_WEEK = argint("steps-per-week", 3)
const LR = argfloat("lr", 0.2)
const MULTIPLIER = argfloat("multiplier", 1500.0)
const OUTDIR = string(argval("out", joinpath(OUT_ROOT, "real_data")))
const PARALLEL = Symbol(argval("parallel", "runs"))

banner("Inverse PDE fit on real satellite data")
report_threads()
println("sites            : ", join(SITES, ", "))
println("restarts         : ", NMODELS)
println("epochs           : ", EPOCHS)
println("steps per week   : ", STEPS_PER_WEEK, "  (", 52 * STEPS_PER_WEEK, " steps/year)")
println("learning rate    : ", LR)
println("output           : ", OUTDIR)

# ---------------------------------------------------------------------------
banner("Loading data")
sites = load_sites(DATA_DIR, SITES; multiplier = MULTIPLIER, T = Float64)
for s in sites
    ys = years(s)
    @printf("  %-12s %2d years %d-%d  %dx%d  precip %.0f-%.0f mm\n",
            s.name, length(s), minimum(ys), maximum(ys), size(s)...,
            minimum(o.precipitation for o in s), maximum(o.precipitation for o in s))
end

stats = biomass_stats(sites)
@printf("  biomass (per-image mean): %.2f - %.2f, mean %.2f\n",
        stats.min_biomass, stats.max_biomass, stats.mean_biomass)

cfg = SimConfig(steps_per_week = STEPS_PER_WEEK, year_time_units = 1.0)
problem = InverseProblem(sites, cfg; average = true)
println("\n", problem)

# The fitted diffusion coefficients for this dataset land around 18-34, so the
# explicit-Euler bound is a live constraint rather than a formality.
limit = diffusion_stability_limit(cfg)
@printf("diffusion stability limit at %d steps/week: %.1f\n", STEPS_PER_WEEK, limit)
limit < 35 && @warn """the stability limit is below the diffusion coefficients this
                       dataset typically fits (~18-34). Expect a high divergence
                       rate; --steps-per-week 4 gives a limit of 52.""" limit

mkpath(joinpath(OUTDIR, "runs"))
save_json(joinpath(OUTDIR, "data_info.json"), Dict(
    "sites" => SITES,
    "locations" => [s.name for s in sites],
    "ndvi_to_biomass_multiplier" => MULTIPLIER,
    "use_delta_loss" => true,
    "steps_per_week" => STEPS_PER_WEEK,
    "total_steps_per_year" => 52 * STEPS_PER_WEEK,
    "year_time_units" => cfg.year_time_units,
    "global_stats" => Dict(string(k) => v for (k, v) in pairs(stats)),
    "time_points_per_location" => Dict(s.name => length(s) for s in sites),
))

# ---------------------------------------------------------------------------
banner("Fitting $NMODELS models")

# Seeds follow the Python convention (77 + 102 * index) so run indices line up
# with the existing results, even though the RNG streams themselves differ.
seeds = [77 + 102 * i for i in 0:(NMODELS - 1)]
traincfg = TrainConfig(epochs = EPOCHS, learning_rate = LR, lr_decay = 0.999,
                       grad_clip = 10.0, save_interval = 10, print_interval = 100,
                       verbose = PARALLEL !== :runs)

t0 = time()
results = train_many(problem, seeds; cfg = traincfg, parallel = PARALLEL)
elapsed = time() - t0
@printf("\nAll runs finished in %.1f min\n", elapsed / 60)

# ---------------------------------------------------------------------------
banner("Results")
for (i, r) in enumerate(results)
    save_run(joinpath(OUTDIR, "runs", @sprintf("run_%02d.json", i - 1)), r;
             metadata = Dict("run_id" => i - 1, "sites" => SITES,
                             "steps_per_week" => STEPS_PER_WEEK))
    @printf("  run %2d  seed %5d  %5d/%d epochs  final loss %12.4f  %s  %6.1f min\n",
            i - 1, r.seed, r.epochs_run, EPOCHS, r.final_loss,
            r.converged ? "ok      " : "DIVERGED", r.elapsed_seconds / 60)
end

ok = findall(r -> r.converged && isfinite(r.final_loss), results)
if isempty(ok)
    @error """every run diverged. All runs are still written to $(joinpath(OUTDIR, "runs")).
              A high divergence rate is normal for this problem — the published Python
              results discard about a third of runs — but if it is *every* run, check
              the diffusion stability limit above against --steps-per-week."""
    exit(1)
end

losses = [results[i].final_loss for i in ok]
@printf("\n%d/%d runs usable.  final loss: mean %.4f  std %.4f  min %.4f  max %.4f\n",
        length(ok), length(results), Statistics.mean(losses),
        Statistics.std(losses), minimum(losses), maximum(losses))

# Pooled parameter table, in the same format the analysis scripts read.
B = mean_training_biomass(DATA_DIR, SITES; multiplier = MULTIPLIER)
params = [results[i].params for i in ok]
csv = write_parameter_table(joinpath(OUTDIR, "final_parameter_values.csv"), ok .- 1, params;
                            extra = Dict("turing_value" => turing_value.(params),
                                         "composite_value" => [composite_value(p, B) for p in params],
                                         "final_loss" => losses))
println("parameter table -> ", csv)

best = ok[argmin(losses)]
println("\nBest run: ", best - 1, "  (loss ", @sprintf("%.4f", results[best].final_loss), ")")
show(stdout, MIME"text/plain"(), results[best].params)
@printf("turing value    %.6f   %s\n", turing_value(results[best].params),
        turing_value(results[best].params) < 0 ? "(patterning possible)" : "(no Turing instability)")

# ---------------------------------------------------------------------------
banner("Plotting")
plt = plot(xlabel = "epoch", ylabel = "delta-MSE loss", yscale = :log10,
           title = "Real-data fits ($(length(ok)) runs)", legend = false,
           size = (900, 550), dpi = 150)
for i in ok
    plot!(plt, max.(results[i].loss_history, eps()), lw = 1, alpha = 0.6, color = :steelblue)
end
plot!(plt, max.(results[best].loss_history, eps()), lw = 2.5, color = :black)
png_path = joinpath(OUTDIR, "loss_curves.png")
savefig(plt, png_path)
savefig(plt, joinpath(OUTDIR, "loss_curves.pdf"))
println("  ", png_path)

println("\nDone. Outputs in ", OUTDIR)
