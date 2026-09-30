#!/usr/bin/env julia
"""
Adam-only fitting in log-parameters.

Why this experiment exists. The published ensemble (Adam, raw parameters, 7500
epochs) agrees on all nine coefficients to CV 0.018-0.226. The converged L-BFGS
ensemble in `julia/results/hybrid_realdata` reaches the *same loss* — median
739.59 against 739.71 — with *equally small gradients* and CV up to 1.45, spanning
five orders of magnitude on the water coefficients. So the published agreement is
not a statement about what the data determines; it is a statement about where
Adam's dynamics come to rest.

That leaves a question this script answers: is the clustering caused by the raw
parametrisation? Adam normalises each coordinate by its own gradient history, so
its steps are scale-free in update magnitude but not in *relative* magnitude — the
same absolute step is a rounding error for a coefficient near 30 and a total
rewrite for one near 0.01. Optimising `u = log θ` makes the step relative. If the
clustering survives that change it is a property of Adam's trajectory; if it
dissolves, the raw parametrisation was doing the constraining.

The raw-parameter arm is *not* re-run here: the published 47-fit ensemble is that
arm, it uses this exact schedule, and the port reproduces it to 2e-8. Compare
against `results/parameter_history_analysis/four_site_final_parameter_values.csv`.

Usage
-----
    julia --project=julia -t auto julia/scripts/train_adam_logspace.jl \\
        --models 6 --epochs 7500 --concurrency 6

Options
    --models 6            random restarts
    --epochs 7500         Adam epochs per model (the published schedule)
    --raw                 optimise theta instead of log theta (control arm)
    --concurrency 6       models in flight at once
    --lr 0.2              learning rate
    --report-every 3      print a parameter table every N completions
    --out <dir>

Budgeting
---------
One epoch is one gradient evaluation, and this machine sustains ~0.64 of them per
second in total regardless of concurrency (the loss is memory-bandwidth bound).
So wall clock is `models * epochs / 0.64` seconds and nothing else — 6 models at
7500 epochs is ~19.5 h. The script prints the estimate before starting.
"""

using InverseTuring
using Printf
using Random
import Statistics, CSV, DataFrames, JSON

include(joinpath(@__DIR__, "common.jl"))

const NMODELS = argint("models", 6)
const EPOCHS = argint("epochs", 7500)
const LOGSPACE = !("--raw" in ARGS)
const SITES = arglist("sites", ["b", "i", "c", "e"])
const STEPS_PER_WEEK = argint("steps-per-week", 3)
const LR = argfloat("lr", 0.2)
const MULTIPLIER = argfloat("multiplier", 1500.0)
const CONCURRENCY = argint("concurrency", 6)
const REPORT_EVERY = argint("report-every", 3)
const OUTDIR = string(argval("out", joinpath(OUT_ROOT,
                      LOGSPACE ? "adam_logspace" : "adam_raw")))
# ~250 trajectory rows per model whatever the epoch count, so the CSVs stay a
# readable size and the plotter gets a smooth curve.
const SAVE_INTERVAL = max(1, EPOCHS ÷ 250)
const TRAJDIR = joinpath(OUTDIR, "trajectories")
const EVAL_RATE = 0.64

banner("Adam-only fit, $(LOGSPACE ? "log" : "raw") parameters")
report_threads()
@printf("models %d | epochs %d | lr %.3g | steps/week %d | snapshot every %d\n",
        NMODELS, EPOCHS, LR, STEPS_PER_WEEK, SAVE_INTERVAL)
@printf("budget: %d gradient evaluations -> estimated %.1f h at %.2f evals/s\n",
        NMODELS * EPOCHS, NMODELS * EPOCHS / EVAL_RATE / 3600, EVAL_RATE)
println("output: ", OUTDIR)

sites = load_sites(DATA_DIR, SITES; multiplier = MULTIPLIER, T = Float64)
cfg = SimConfig(steps_per_week = STEPS_PER_WEEK, year_time_units = 1.0)
prob = InverseProblem(sites, cfg; threaded = false)   # models are the parallel axis
mkpath(joinpath(OUTDIR, "runs"))
mkpath(TRAJDIR)

# ---------------------------------------------------------------------------
# Screen starts the same way the hybrid run does, so the two ensembles are drawn
# from the same pool of seeds and the comparison is not confounded by which
# random starts each got.
banner("Screening seeds")
function screen_seeds(n)
    accepted, rejected = Int[], Int[]
    idx = 0
    while length(accepted) < n && idx < 20 * n
        s = 77 + 102 * idx
        idx += 1
        isfinite(loss(paramvector(randparams(Xoshiro(s))), prob)) ?
            push!(accepted, s) : push!(rejected, s)
    end
    return accepted, rejected
end
seeds, rejected = screen_seeds(NMODELS)
length(seeds) == NMODELS || error("only found $(length(seeds)) usable seeds")
@printf("accepted %d, rejected %d unstable\nseeds: %s\n",
        length(seeds), length(rejected), join(seeds, ", "))

const TRAIN_CFG = TrainConfig(epochs = EPOCHS, learning_rate = LR, lr_decay = 0.999,
                              grad_clip = 10.0, save_interval = SAVE_INTERVAL,
                              verbose = false, logspace = LOGSPACE)

"""
Fit one model, streaming its trajectory as it goes.

Rows are written and flushed from `train`'s callback rather than assembled from
`TrainResult` afterwards, so a run in progress can be inspected and plotted. The
first version of this script built the file at the end, which meant four hours
with an empty output directory.
"""
function fit_one(seed::Int)
    θ0 = paramvector(randparams(Xoshiro(seed)))
    phase = LOGSPACE ? "adam-log" : "adam-raw"
    io = open(joinpath(TRAJDIR, @sprintf("seed_%05d_trajectory.csv", seed)), "w")
    println(io, "phase,step,t,loss,gnorm," * join(PARAM_NAMES, ","))
    println(io, "init,0,0.0,", loss(θ0, prob), ",NaN,", join(θ0, ","))
    flush(io)

    t0 = time()
    res = try
        train(prob, θ0; cfg = TRAIN_CFG, seed = seed,
              callback = (epoch, θ, l, g) -> begin
                  println(io, phase, ",", epoch, ",", time() - t0, ",", l, ",", g,
                          ",", join(θ, ","))
                  flush(io)
              end)
    catch e
        close(io)
        rethrow()
    end

    θf = paramvector(res.params)
    lf, gf = loss_and_gradient(prob, θf)          # raw-space gradient, comparable
                                                  # across both parametrisations
    println(io, "final,", res.epochs_run, ",", res.elapsed_seconds, ",", lf, ",",
            sqrt(sum(abs2, gf)), ",", join(θf, ","))
    close(io)
    return (seed = seed, ok = res.converged && isfinite(lf), theta = θf, loss = lf,
            gnorm = sqrt(sum(abs2, gf)),
            gnorm_opt = isempty(res.gradnorm_history) ? NaN : res.gradnorm_history[end],
            epochs = res.epochs_run, converged = res.converged,
            seconds = res.elapsed_seconds,
            best_loss = isempty(res.loss_history) ? NaN :
                        minimum(filter(isfinite, res.loss_history)))
end

# ---------------------------------------------------------------------------
banner("Fitting $NMODELS models")
results = Vector{Any}(undef, NMODELS)
tstart = time()
const PROGRESS_CSV = joinpath(OUTDIR, "progress.csv")
const reportlock = ReentrantLock()
const finished = Ref(0)
const order = Int[]

open(PROGRESS_CSV, "w") do io
    println(io, "finish_order,model_id,seed,ok,loss,best_loss,gnorm,gnorm_optspace," *
                "epochs,minutes,converged," * join(PARAM_NAMES, ","))
end

short_names = [replace(n, "_coeff" => "", "_rate" => "", "_" => " ") for n in PARAM_NAMES]

function batch_report(m)
    recent = order[max(1, end - m + 1):end]
    row(label, vals, fmt) =
        "RPT  " * rpad(label, 26) * join((lpad(Printf.format(fmt, v), 13) for v in vals))
    println("\nRPT ", "=" ^ 96)
    @printf("RPT  PARAMETERS AFTER %d/%d MODELS   (%.1f min elapsed)\n",
            finished[], NMODELS, (time() - tstart) / 60)
    println("RPT ", "-" ^ 96)
    println("RPT  ", rpad("parameter", 26), join((lpad("seed $(results[k].seed)", 13) for k in recent)))
    g5 = Printf.Format("%.5g")
    for (pi, nm) in enumerate(short_names)
        println(row(nm, [results[k].theta[pi] for k in recent], g5))
    end
    println(row("final loss", [results[k].loss for k in recent], Printf.Format("%.3f")))
    println(row("final ||grad||", [results[k].gnorm for k in recent], Printf.Format("%.3g")))
    good = [results[k] for k in order if results[k].ok]
    if length(good) >= 2
        cvv(v) = Statistics.std(v) / abs(Statistics.mean(v))
        @printf("RPT  running (n=%d): loss %.3f-%.3f | CV l3 %.3f, d3 %.3f, l1 %.3f, j %.3f\n",
                length(good), extrema([r.loss for r in good])...,
                cvv([r.theta[6] for r in good]), cvv([r.theta[3] for r in good]),
                cvv([r.theta[4] for r in good]), cvv([r.theta[9] for r in good]))
    end
    println("RPT ", "=" ^ 96)
    flush(stdout)
end

queue = Channel{Int}(NMODELS)
for k in 1:NMODELS
    put!(queue, k)
end
close(queue)

@sync for _ in 1:CONCURRENCY
    Threads.@spawn for k in queue
        results[k] = try
            fit_one(seeds[k])
        catch e
            (seed = seeds[k], ok = false, theta = fill(NaN, NPARAMS), loss = NaN,
             gnorm = NaN, gnorm_opt = NaN, epochs = 0, converged = false,
             seconds = 0.0, best_loss = NaN)
        end
        r = results[k]
        lock(reportlock) do
            finished[] += 1
            push!(order, k)
            open(PROGRESS_CSV, "a") do io
                println(io, finished[], ",", k - 1, ",", r.seed, ",", r.ok, ",", r.loss,
                        ",", r.best_loss, ",", r.gnorm, ",", r.gnorm_opt, ",", r.epochs,
                        ",", r.seconds / 60, ",", r.converged, ",", join(r.theta, ","))
            end
            @printf("[%2d/%2d] seed %5d  loss %.3f (best seen %.3f)  |g| %.3g  %d epochs  %.0f min\n",
                    finished[], NMODELS, r.seed, r.loss, r.best_loss, r.gnorm,
                    r.epochs, r.seconds / 60)
            flush(stdout)
            finished[] % REPORT_EVERY == 0 && batch_report(REPORT_EVERY)
        end
    end
end
finished[] % REPORT_EVERY == 0 || batch_report(min(REPORT_EVERY, finished[]))
@printf("\nAll models finished in %.1f min wall clock\n", (time() - tstart) / 60)

# ---------------------------------------------------------------------------
banner("Results")
ok = filter(r -> r.ok, results)
if isempty(ok)
    @error "every model failed"
else
    @printf("%d/%d usable | loss %.3f-%.3f | ‖∇L‖ %.3g-%.3g\n", length(ok), NMODELS,
            extrema([r.loss for r in ok])..., extrema([r.gnorm for r in ok])...)
    pinned = count(r -> any(v -> v <= 1.1e-4, r.theta), ok)
    @printf("models with >=1 parameter at the 1e-4 floor: %d/%d\n", pinned, length(ok))
    cvv(v) = Statistics.std(v) / abs(Statistics.mean(v))
    @printf("\n%-30s %10s %8s %12s\n", "parameter", "median", "CV", "max/min")
    for (i, n) in enumerate(PARAM_NAMES)
        v = [r.theta[i] for r in ok]
        @printf("%-30s %10.4g %8.3f %12.4g\n", n, Statistics.median(v), cvv(v),
                maximum(v) / minimum(v))
    end

    for (k, r) in enumerate(results)
        save_json(joinpath(OUTDIR, "runs", @sprintf("model_%02d.json", k - 1)), Dict(
            "model_id" => k - 1, "seed" => r.seed, "ok" => r.ok,
            "logspace" => LOGSPACE, "epochs" => r.epochs, "params" => r.theta,
            "loss" => r.loss, "best_loss" => r.best_loss, "gradnorm" => r.gnorm,
            "gradnorm_optspace" => r.gnorm_opt, "converged" => r.converged,
            "seconds" => r.seconds, "param_names" => collect(PARAM_NAMES)))
    end
    ids = [k - 1 for (k, r) in enumerate(results) if r.ok]
    params = [RietkerkParams(r.theta) for r in ok]
    B = mean_training_biomass(DATA_DIR, SITES; multiplier = MULTIPLIER)
    write_parameter_table(joinpath(OUTDIR, "final_parameter_values.csv"), ids, params;
                          extra = Dict("turing_value" => turing_value.(params),
                                       "composite_value" => [composite_value(p, B) for p in params],
                                       "final_loss" => [r.loss for r in ok],
                                       "final_gradnorm" => [r.gnorm for r in ok]))
end
println("\nDone. Outputs in ", OUTDIR)
