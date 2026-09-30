#!/usr/bin/env julia
"""
Hybrid Adam -> L-BFGS fitting on real satellite data.

Rationale, from the optimiser comparison in `julia/results/lbfgs_trajectories`:

* **Adam alone** needs ~7500 epochs (~9 h/model here, 25 h on the A100) and still
  stops at `‖∇L‖` between 2 and 353 — the schedule runs out before the gradient
  does, and the stopping quality varies ~180x between runs.
* **L-BFGS alone** converges far harder (one seed reached `‖∇L‖ = 4e-5` in 631
  evaluations) but dives straight down the objective's flat directions and pins
  parameters against the `[1e-4, 1e4]` clamp — solutions the project's own
  `drop_degenerate` filter would reject.
* **Adam then L-BFGS** uses Adam's per-parameter normalisation to cross the badly
  scaled early landscape without touching the bounds, then hands a good interior
  point to L-BFGS to finish.

The L-BFGS phase optimises `u = log θ`: positivity is automatic and it removes the
three orders of magnitude of scale spread between the diffusion and rate constants.
Bounds are `log([1e-4, 1e4])`, the same feasible set Adam's post-step clamp enforces.

Usage
-----
    julia --project=julia -t auto julia/scripts/train_hybrid.jl --models 30

Options
    --models 30           number of random restarts
    --adam-epochs 100     Adam warm-up epochs per model
    --lbfgs-evals 400     L-BFGS loss+gradient evaluations per model
    --sites b,i,c,e       training subsites
    --steps-per-week 3    Euler sub-steps per forcing week
    --lr 0.2              Adam learning rate
    --out <dir>

Outputs
-------
    progress.csv          one row per model, appended as each finishes
    trajectories/         seed_NNNNN_trajectory.csv, one row per Adam snapshot
                          and per L-BFGS evaluation; plot with
                          `julia/scripts/plot_lbfgs_trajectories.jl`
    runs/model_NN.json, summary.csv, final_parameter_values.csv

Budgeting
---------
`--lbfgs-evals` caps *work*, not wall-clock, deliberately. This machine saturates
at ~0.64 gradient evaluations per second in total, almost independently of how
many models run concurrently — 4 concurrent gives 0.56/s, 30 gives 0.64/s, because
the loss is memory-bandwidth bound. A wall-clock cap under that load would
silently hand each model ~30 iterations instead of the ~600 it needs.

Estimate wall time as `models * (adam_epochs + lbfgs_evals) / 0.64` seconds; the
script prints this at startup. Re-measure the rate on a different machine.

Models run concurrently, one thread each, sharing the loaded rasters. Start Julia
with `-t auto`.
"""

using InverseTuring
using Optim
using Printf
using Random
import Statistics, CSV, DataFrames, JSON

include(joinpath(@__DIR__, "common.jl"))

const NMODELS = argint("models", 30)
const ADAM_EPOCHS = argint("adam-epochs", 100)
const LBFGS_EVALS = argint("lbfgs-evals", 400)
const SITES = arglist("sites", ["b", "i", "c", "e"])
const STEPS_PER_WEEK = argint("steps-per-week", 3)
const LR = argfloat("lr", 0.2)
const MULTIPLIER = argfloat("multiplier", 1500.0)
const OUTDIR = string(argval("out", joinpath(OUT_ROOT, "hybrid_realdata")))
const LO, HI = 1e-4, 1e4
const LOGLO, LOGHI = log(LO), log(HI)
const SAVE_INTERVAL = max(1, ADAM_EPOCHS ÷ 20)

# Per-model parameter trajectories. Only the endpoint used to be kept, which made
# it impossible to tell a converged fit from one still drifting along a flat
# direction — the distinction this project turns on. Rows are appended and
# flushed as they are produced, so a crash or an interrupt keeps everything up to
# that point, and a run in progress can be plotted.
const TRAJDIR = joinpath(OUTDIR, "trajectories")

banner("Hybrid Adam -> L-BFGS fit on real data")
report_threads()
const EVAL_RATE = 0.64          # measured gradient evals/second, whole machine
@printf("models %d | Adam %d epochs | L-BFGS %d evals | steps/week %d\n",
        NMODELS, ADAM_EPOCHS, LBFGS_EVALS, STEPS_PER_WEEK)
@printf("budget: %d evaluations total -> estimated %.1f h at %.2f evals/s\n",
        NMODELS * (ADAM_EPOCHS + LBFGS_EVALS),
        NMODELS * (ADAM_EPOCHS + LBFGS_EVALS) / EVAL_RATE / 3600, EVAL_RATE)
println("output: ", OUTDIR)

sites = load_sites(DATA_DIR, SITES; multiplier = MULTIPLIER, T = Float64)
cfg = SimConfig(steps_per_week = STEPS_PER_WEEK, year_time_units = 1.0)
prob = InverseProblem(sites, cfg; threaded = false)   # models are the parallel axis
Bmax = maximum(maximum(o.biomass) for s in sites for o in s.observations)
mkpath(joinpath(OUTDIR, "runs"))
mkpath(TRAJDIR)

# ---------------------------------------------------------------------------
# Seed screening. Some Uniform[0,1) starts sit outside the explicit scheme's
# stability region and give a non-finite loss immediately; the project handles
# this with find_stable_seed.py. A finite initial loss is the cheap proxy.
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
@printf("accepted %d, rejected %d unstable (%.0f%%)\n",
        length(seeds), length(rejected),
        100 * length(rejected) / (length(seeds) + length(rejected)))

# ---------------------------------------------------------------------------
"""Thrown out of the objective once a model's L-BFGS evaluation budget is spent."""
struct EvalLimit <: Exception end

"""
Adam warm-up followed by bounded L-BFGS in log-parameters.

If the Adam phase diverges — which it does on a substantial fraction of seeds at
`lr = 0.2`, matching the ~1/3 failure rate of the published runs — the model is
**not** discarded. L-BFGS is restarted from the original random point instead.
The earlier optimiser comparison showed L-BFGS handling starts that Adam cannot
(seed 77 diverged under Adam here yet reached `‖∇L‖ = 4e-5` under L-BFGS alone),
so throwing those away would lose good models to a warm-up that was supposed to
help. The `stage` field records which path each model took.
"""
function hybrid_fit(seed::Int)
    θ0 = paramvector(randparams(Xoshiro(seed)))

    traj = open(joinpath(TRAJDIR, @sprintf("seed_%05d_trajectory.csv", seed)), "w")
    println(traj, "phase,step,t,loss,gnorm," * join(PARAM_NAMES, ","))
    trow(phase, step, t, l, g, θ) =
        (println(traj, phase, ",", step, ",", t, ",", l, ",", g, ",", join(θ, ","));
         flush(traj))

    t0 = time()
    trow("init", 0, 0.0, loss(θ0, prob), NaN, θ0)
    warm = train(prob, θ0;
                 cfg = TrainConfig(epochs = ADAM_EPOCHS, learning_rate = LR,
                                   lr_decay = 0.999, grad_clip = 10.0,
                                   save_interval = SAVE_INTERVAL,
                                   verbose = false),
                 seed = seed)
    t_adam = time() - t0

    # Adam snapshots are taken every SAVE_INTERVAL epochs (matching the Python
    # loop), while the loss is recorded every epoch; index defensively because
    # the snapshot list carries one extra entry when the epoch count is not a
    # multiple of the interval.
    for (i, snap) in enumerate(warm.parameter_history)
        ep = min((i - 1) * SAVE_INTERVAL + 1, length(warm.loss_history))
        ep >= 1 || continue
        trow("adam", ep, t_adam * ep / max(1, ADAM_EPOCHS), warm.loss_history[ep],
             NaN, [snap[n] for n in PARAM_NAMES])
    end
    θ_adam = paramvector(warm.params)
    loss_adam = warm.converged ? loss(θ_adam, prob) : NaN
    adam_ok = warm.converged && isfinite(loss_adam)
    g_adam = adam_ok ? loss_and_gradient(prob, θ_adam)[2] : fill(NaN, NPARAMS)

    # Fall back to the random start rather than refining a diverged point.
    θ_start = adam_ok ? θ_adam : θ0
    stage = adam_ok ? "adam+lbfgs" : "lbfgs-only (adam diverged)"

    # The evaluation cap is enforced inside the objective, and the best point is
    # tracked here rather than taken from Optim. `Fminbox`'s `f_calls_limit`
    # applies per inner solve, so the outer barrier loop simply restarts and the
    # run never terminates; throwing out of the objective is unambiguous.
    nev = Ref(0)
    best_l = Ref(Inf)
    best_θ = Ref(copy(θ_start))

    function fg!(F, G, u)
        nev[] >= LBFGS_EVALS && throw(EvalLimit())
        θ = exp.(clamp.(u, LOGLO, LOGHI))
        l, gθ = loss_and_gradient(prob, θ)
        G === nothing || (G .= gθ .* θ)         # chain rule dL/du = dL/dθ * θ
        nev[] += 1
        if isfinite(l) && l < best_l[]
            best_l[] = l
            best_θ[] = copy(θ)
        end
        trow("lbfgs", nev[], time() - t0, l, sqrt(sum(abs2, gθ)), θ)
        return l
    end

    t1 = time()
    converged = false
    try
        res = optimize(Optim.NLSolversBase.only_fg!(fg!),
                       fill(LOGLO, NPARAMS), fill(LOGHI, NPARAMS),
                       clamp.(log.(clamp.(θ_start, LO, HI)), LOGLO, LOGHI),
                       Fminbox(LBFGS()),
                       Optim.Options(iterations = 100_000, g_tol = 1e-8))
        converged = Optim.converged(res)
    catch e
        e isa EvalLimit || rethrow()            # hit the budget, keep the best point
    end
    t_lbfgs = time() - t1

    θf = best_θ[]
    lf, gf = loss_and_gradient(prob, θf)
    trow("final", nev[], time() - t0, lf, sqrt(sum(abs2, gf)), θf)
    close(traj)

    return (seed = seed, ok = isfinite(lf), stage = stage,
            theta_adam = θ_adam, loss_adam = loss_adam,
            gnorm_adam = adam_ok ? sqrt(sum(abs2, g_adam)) : NaN,
            theta = θf, loss = lf, gnorm = sqrt(sum(abs2, gf)),
            adam_epochs = warm.epochs_run, lbfgs_evals = nev[],
            t_adam = t_adam, t_lbfgs = t_lbfgs, converged = converged)
end

# ---------------------------------------------------------------------------
banner("Fitting $NMODELS models")
results = Vector{Any}(undef, NMODELS)
tstart = time()

# Incremental reporting. Every completed model is appended to progress.csv
# immediately, so a crash or an interrupt keeps everything finished so far; and
# every REPORT_EVERY completions a full parameter table is printed. Report lines
# are prefixed "RPT " so a log watcher can filter for them.
const REPORT_EVERY = argint("report-every", 4)
const PROGRESS_CSV = joinpath(OUTDIR, "progress.csv")
const reportlock = ReentrantLock()
const finished = Ref(0)
const order = Int[]                       # completion order, for batch reports

open(PROGRESS_CSV, "w") do io
    println(io, "finish_order,model_id,seed,ok,adam_loss,hybrid_loss,adam_gnorm," *
                "hybrid_gnorm,lbfgs_evals,minutes,converged," * join(PARAM_NAMES, ","))
end

short_names = [replace(n, "_coeff" => "", "_rate" => "", "_" => " ") for n in PARAM_NAMES]

"""
Print the final parameters of the `m` most recently finished models.

Column count varies (a partial final batch, or a different `--report-every`), so
the rows are assembled by string concatenation rather than a fixed `@printf`
format — a hard-coded arity here would throw after hours of compute.
"""
function batch_report(m)
    recent = order[max(1, end - m + 1):end]
    row(label, vals, fmt) =
        "RPT  " * rpad(label, 26) * join((lpad(Printf.format(fmt, v), 13) for v in vals))
    println("\nRPT ", "=" ^ 96)
    @printf("RPT  PARAMETERS AFTER %d/%d MODELS   (%.1f min elapsed)\n",
            finished[], NMODELS, (time() - tstart) / 60)
    println("RPT ", "-" ^ 96)
    println("RPT  ", rpad("parameter", 26),
            join((lpad("model $(k - 1)", 13) for k in recent)))
    g5 = Printf.Format("%.5g")
    f3 = Printf.Format("%.3f")
    g3 = Printf.Format("%.3g")
    for (pi, nm) in enumerate(short_names)
        println(row(nm, [results[k].theta[pi] for k in recent], g5))
    end
    println(row("final loss", [results[k].loss for k in recent], f3))
    println(row("final ||grad||", [results[k].gnorm for k in recent], g3))
    println(row("stage", [startswith(results[k].stage, "adam+") ? "hybrid" : "lbfgs-only"
                          for k in recent], Printf.Format("%s")))

    # Running spread over every model finished so far — the identifiability signal.
    good = [results[k] for k in order if results[k].ok]
    if length(good) >= 2
        cvv(v) = Statistics.std(v) / abs(Statistics.mean(v))
        losses = [r.loss for r in good]
        @printf("RPT  running (n=%d): loss %.3f-%.3f | CV mortality %.3f, biomass-diff %.3f, evap %.3f, wue %.3f\n",
                length(good), minimum(losses), maximum(losses),
                cvv([r.theta[6] for r in good]), cvv([r.theta[3] for r in good]),
                cvv([r.theta[4] for r in good]), cvv([r.theta[9] for r in good]))
    end
    println("RPT ", "=" ^ 96)
    flush(stdout)
end

# Bounded concurrency rather than all-at-once. Total throughput is flat in the
# number of concurrent models (the loss is memory-bandwidth bound), so running
# fewer at a time costs nothing overall but lets each finish sooner — which is
# what makes the every-N progress reports arrive spread out instead of all at the
# end.
const CONCURRENCY = argint("concurrency", 6)
@printf("running %d models at a time\n\n", CONCURRENCY)
queue = Channel{Int}(NMODELS)
for k in 1:NMODELS
    put!(queue, k)
end
close(queue)

@sync for _ in 1:CONCURRENCY
    Threads.@spawn for k in queue
    results[k] = try
        hybrid_fit(seeds[k])
    catch e
        (seed = seeds[k], ok = false, stage = "error: " * first(sprint(showerror, e), 80),
         theta_adam = fill(NaN, NPARAMS), loss_adam = NaN, gnorm_adam = NaN,
         theta = fill(NaN, NPARAMS), loss = NaN, gnorm = NaN,
         adam_epochs = 0, lbfgs_evals = 0, t_adam = 0.0, t_lbfgs = 0.0,
         converged = false)
    end
    r = results[k]
    lock(reportlock) do
        finished[] += 1
        push!(order, k)
        open(PROGRESS_CSV, "a") do io
            println(io, finished[], ",", k - 1, ",", r.seed, ",", r.ok, ",",
                    r.loss_adam, ",", r.loss, ",", r.gnorm_adam, ",", r.gnorm, ",",
                    r.lbfgs_evals, ",", (r.t_adam + r.t_lbfgs) / 60, ",", r.converged,
                    ",", join(r.theta, ","))
        end
        @printf("[%2d/%2d] seed %5d  Adam %.2f -> L-BFGS %.2f   |g| %.3g -> %.3g   %.0f min\n",
                finished[], NMODELS, r.seed, r.loss_adam, r.loss, r.gnorm_adam, r.gnorm,
                (r.t_adam + r.t_lbfgs) / 60)
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
isempty(ok) && (@error "every model failed"; exit(1))
@printf("%d/%d usable\n", length(ok), NMODELS)
@printf("loss  : Adam %.3f-%.3f  ->  hybrid %.3f-%.3f\n",
        extrema([r.loss_adam for r in ok])..., extrema([r.loss for r in ok])...)
@printf("|grad|: Adam %.3g-%.3g  ->  hybrid %.3g-%.3g\n",
        extrema([r.gnorm_adam for r in ok])..., extrema([r.gnorm for r in ok])...)
@printf("L-BFGS improved the loss in %d/%d models\n",
        count(r -> r.loss < r.loss_adam, ok), length(ok))

# Bound-pinning: the failure mode L-BFGS is prone to, and what the project's
# drop_degenerate filter screens on.
pinned = [count(v -> v <= 1.1 * LO, r.theta) for r in ok]
@printf("models with >=1 parameter pinned at the %.0e floor: %d/%d\n",
        LO, count(>(0), pinned), length(ok))

for (k, r) in enumerate(results)
    save_json(joinpath(OUTDIR, "runs", @sprintf("model_%02d.json", k - 1)), Dict(
        "model_id" => k - 1, "seed" => r.seed, "ok" => r.ok, "stage" => r.stage,
        "adam" => Dict("params" => r.theta_adam, "loss" => r.loss_adam,
                       "gradnorm" => r.gnorm_adam, "epochs" => r.adam_epochs,
                       "seconds" => r.t_adam),
        "hybrid" => Dict("params" => r.theta, "loss" => r.loss, "gradnorm" => r.gnorm,
                         "evals" => r.lbfgs_evals, "seconds" => r.t_lbfgs,
                         "converged" => r.converged),
        "param_names" => collect(PARAM_NAMES)))
end

ids = [k - 1 for (k, r) in enumerate(results) if r.ok]
params = [RietkerkParams(r.theta) for r in ok]
B = mean_training_biomass(DATA_DIR, SITES; multiplier = MULTIPLIER)
write_parameter_table(joinpath(OUTDIR, "final_parameter_values.csv"), ids, params;
                      extra = Dict("turing_value" => turing_value.(params),
                                   "composite_value" => [composite_value(p, B) for p in params],
                                   "final_loss" => [r.loss for r in ok],
                                   "final_gradnorm" => [r.gnorm for r in ok],
                                   "adam_loss" => [r.loss_adam for r in ok]))

CSV.write(joinpath(OUTDIR, "summary.csv"), DataFrames.DataFrame(
    model_id = [k - 1 for k in 1:NMODELS], seed = [r.seed for r in results],
    ok = [r.ok for r in results], stage = [r.stage for r in results],
    adam_loss = [r.loss_adam for r in results], hybrid_loss = [r.loss for r in results],
    adam_gnorm = [r.gnorm_adam for r in results], hybrid_gnorm = [r.gnorm for r in results],
    lbfgs_evals = [r.lbfgs_evals for r in results],
    minutes = [(r.t_adam + r.t_lbfgs) / 60 for r in results],
    converged = [r.converged for r in results]))

# ---------------------------------------------------------------------------
banner("Identifiability across the ensemble")
# The parameters that touch the observed biomass field directly should be far
# better determined than those reaching it only through unobserved water.
cv(v) = Statistics.std(v) / abs(Statistics.mean(v))
vals = Dict(n => [getfield(p, i) for p in params] for (i, n) in enumerate(PARAM_NAMES))
@printf("%-32s %10s %12s\n", "parameter", "CV", "max/min")
for (nm, c) in sort([(n, cv(vals[n])) for n in PARAM_NAMES], by = x -> x[2])
    v = vals[nm]
    @printf("%-32s %10.3f %12.4g\n", nm, c, maximum(v) / max(minimum(v), eps()))
end
d1 = vals["surface_water_diffusion_coeff"]; d2 = vals["soil_water_diffusion_coeff"]
r1 = vals["plant_uptake_rate"]; j = vals["water_use_efficiency"]
@printf("\ncombinations:  CV(d1+d2) = %.3f   CV(j*r1) = %.3f   CV(composite) = %.3f\n",
        cv(d1 .+ d2), cv(j .* r1), cv([composite_value(p, B) for p in params]))

println("\nDone. Outputs in ", OUTDIR)
