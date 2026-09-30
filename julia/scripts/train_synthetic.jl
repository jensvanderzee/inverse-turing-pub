#!/usr/bin/env julia
"""
Recover known parameters from synthetic data.

Port of `train_invPDE_synthetic_batch.py` and its single-site variant. Ground-truth
parameters generate noisy observations, the fit tries to recover them, and the
script reports how close it got — the identifiability check that underpins the
real-data results.

Usage
-----
    julia --project=julia -t auto julia/scripts/train_synthetic.jl --preset 4site
    julia --project=julia -t auto julia/scripts/train_synthetic.jl --preset 1site

Options
    --preset 4site|1site  which published experiment to reproduce
    --runs 10             number of random restarts
    --epochs 10000        Adam epochs per restart
    --grid 128            grid edge length
    --seed 42             seed for the ground-truth data (shared by all restarts)
    --out <dir>           output directory
    --parallel runs       :runs | :sites | :none

The two presets differ in more than site count: the 1-site experiment uses
different true parameters, `year_time_units = 1.0` instead of 1.5, and Adam betas
of (0.95, 0.99). Both are reproduced faithfully.
"""

using InverseTuring
using Printf
using Plots
using Random
import Statistics

include(joinpath(@__DIR__, "common.jl"))

const PRESET = string(argval("preset", "4site"))
PRESET in ("4site", "1site") || error("--preset must be 4site or 1site, got $PRESET")

# Configuration of the two published experiments.
const EXPERIMENT = PRESET == "4site" ?
    (truth = SYNTHETIC_TRUTH, n_sites = 4, rain_multiplier = 0.75,
     annual_range = (15.0, 27.0), equilibrium_precip = 21.0, year_time_units = 1.5,
     beta1 = 0.9, beta2 = 0.95) :
    (truth = SYNTHETIC_TRUTH_1SITE, n_sites = 1, rain_multiplier = 1.0,
     annual_range = (16.5, 24.5), equilibrium_precip = 19.0, year_time_units = 1.0,
     beta1 = 0.95, beta2 = 0.99)

const NRUNS = argint("runs", 10)
const EPOCHS = argint("epochs", 10_000)
const GRID = argint("grid", 128)
const SEED = argint("seed", 42)
const OUTDIR = string(argval("out", joinpath(OUT_ROOT, "synthetic_$(PRESET)")))
const PARALLEL = Symbol(argval("parallel", "runs"))

banner("Synthetic parameter-recovery experiment ($PRESET)")
report_threads()
println("grid             : $(GRID)x$(GRID)")
println("sites            : ", EXPERIMENT.n_sites)
println("restarts         : ", NRUNS)
println("epochs           : ", EPOCHS)
println("year time units  : ", EXPERIMENT.year_time_units)
println("output           : ", OUTDIR)

# ---------------------------------------------------------------------------
banner("Generating ground-truth data")
println("True parameters:")
show(stdout, MIME"text/plain"(), EXPERIMENT.truth)
@printf("turing value: %.6f\n", turing_value(EXPERIMENT.truth))

cfg = SimConfig(steps_per_week = 1, year_time_units = EXPERIMENT.year_time_units)
t0 = time()
exp_data = synthetic_experiment(;
    p = EXPERIMENT.truth, grid = (GRID, GRID), n_sites = EXPERIMENT.n_sites,
    rain_multiplier = EXPERIMENT.rain_multiplier, annual_range = EXPERIMENT.annual_range,
    equilibrium_precip = EXPERIMENT.equilibrium_precip, equilibrium_years = 100,
    years = 10, noise_level = 0.05, cfg = cfg, rng = Xoshiro(SEED), T = Float64)
@printf("Equilibrium spin-up + rollouts: %.1f s\n", time() - t0)

eq = exp_data.equilibrium
@printf("Equilibrium biomass: mean %.4f, range %.4f - %.4f\n",
        Statistics.mean(eq), minimum(eq), maximum(eq))
if Statistics.std(eq) < 1e-6
    @warn "equilibrium field is spatially uniform - no Turing pattern formed; " *
          "the inverse problem will be much less identifiable"
end
for (i, t) in enumerate(exp_data.annual_totals)
    @printf("  site %d: annual precipitation %.2f mm, final biomass mean %.4f\n",
            i, t, Statistics.mean(exp_data.targets[i][end]))
end

problem = exp_data.problem
println("\n", problem)

mkpath(joinpath(OUTDIR, "runs"))

# ---------------------------------------------------------------------------
banner("Fitting $NRUNS models")
seeds = [SEED + i for i in 0:(NRUNS - 1)]
traincfg = TrainConfig(epochs = EPOCHS, learning_rate = 0.1, lr_decay = 0.9999,
                       beta1 = EXPERIMENT.beta1, beta2 = EXPERIMENT.beta2,
                       grad_clip = nothing,       # the synthetic script does not clip
                       save_interval = 10, print_interval = 100,
                       verbose = PARALLEL !== :runs)

t0 = time()
results = train_many(problem, seeds; cfg = traincfg, parallel = PARALLEL)
@printf("\nAll runs finished in %.1f min\n", (time() - t0) / 60)

# ---------------------------------------------------------------------------
banner("Parameter recovery")
for (i, r) in enumerate(results)
    save_run(joinpath(OUTDIR, "runs", @sprintf("run_%02d.json", i - 1)), r;
             metadata = Dict("run_id" => i - 1, "preset" => PRESET,
                             "true_params" => paramdict(EXPERIMENT.truth)))
end

ok = findall(r -> r.converged && isfinite(r.final_loss), results)
isempty(ok) && (@error "every run diverged"; exit(1))
losses = [results[i].final_loss for i in ok]
@printf("%d/%d runs usable.  final loss: mean %.6g  min %.6g\n",
        length(ok), length(results), Statistics.mean(losses), minimum(losses))

# How well is each coefficient recovered across restarts?
fitted = Dict(PARAM_NAMES[j] => [paramvector(results[i].params)[j] for i in ok]
              for j in 1:NPARAMS)
tbl = agreement_table(fitted; ground_truth = EXPERIMENT.truth)
println()
show(stdout, MIME"text/plain"(), tbl[:, [:parameter, :mean, :std, :ground_truth,
                                         :mape_vs_gt_pct, :agreement_score]])
println()

B = Statistics.mean(eq)
params = [results[i].params for i in ok]
write_parameter_table(joinpath(OUTDIR, "final_parameter_values.csv"), ok .- 1, params;
                      extra = Dict("turing_value" => turing_value.(params),
                                   "composite_value" => [composite_value(p, B) for p in params],
                                   "final_loss" => losses))

# The composite growth efficiency is the combination the data actually constrains,
# so it should be recovered far better than its individual factors.
true_comp = composite_value(EXPERIMENT.truth, B)
fit_comp = [composite_value(p, B) for p in params]
@printf("\ncomposite value: true %.6f   fitted %.6f +/- %.6f  (%.1f%% error)\n",
        true_comp, Statistics.mean(fit_comp), Statistics.std(fit_comp),
        100 * abs(Statistics.mean(fit_comp) - true_comp) / abs(true_comp))

# ---------------------------------------------------------------------------
banner("Plotting")
plt = plot(xlabel = "epoch", ylabel = "summed delta-MSE", yscale = :log10,
           title = "Synthetic $PRESET recovery", legend = false, size = (900, 550), dpi = 150)
for i in ok
    plot!(plt, max.(results[i].loss_history, eps()), lw = 1, alpha = 0.6, color = :seagreen)
end
savefig(plt, joinpath(OUTDIR, "loss_curves.png"))

# Recovery accuracy per parameter, on a log axis because the coefficients span
# three orders of magnitude.
names_short = [replace(n, "_coeff" => "", "_rate" => "", "_" => " ") for n in PARAM_NAMES]
truevals = paramvector(EXPERIMENT.truth)
meanvals = [Statistics.mean(fitted[n]) for n in PARAM_NAMES]
stdvals = [Statistics.std(fitted[n]) for n in PARAM_NAMES]
plt2 = scatter(1:NPARAMS, truevals, label = "truth", ms = 7, mc = :black,
               xrotation = 40, xticks = (1:NPARAMS, names_short), yscale = :log10,
               ylabel = "value", title = "Parameter recovery ($PRESET)",
               size = (950, 550), dpi = 150)
scatter!(plt2, 1:NPARAMS, max.(meanvals, 1e-6), yerror = stdvals, label = "fitted",
         ms = 5, mc = :orangered)
savefig(plt2, joinpath(OUTDIR, "parameter_recovery.png"))
println("  loss_curves.png, parameter_recovery.png")

println("\nDone. Outputs in ", OUTDIR)
