"""
    TrainConfig

Hyper-parameters for a single fitting run.

Defaults follow `realdata_train_invPDE.py`. The synthetic batch script differs in
three places — `learning_rate = 0.1`, `lr_decay = 0.9999`, `grad_clip = nothing` —
so use [`SYNTHETIC_TRAIN_CONFIG`](@ref) for that experiment rather than editing
these by hand.

`param_min`/`param_max` reproduce the post-step `clamp_(1e-4, 1e4)`, which keeps
the rate constants positive. A parameter pinned at `param_min` at the end of a run
is a failed fit, not a small coefficient — [`tier1_filter`](@ref) exists to catch
exactly that.

`chunk_size` is the ForwardDiff chunk width; see [`DEFAULT_CHUNK`](@ref) for why
the default is not 9.

`logspace` optimises `u = log θ` instead of `θ`. Adam itself is unchanged — same
betas, same epsilon, same decay — so the only difference is the geometry it moves
through. This matters here because the nine coefficients span four orders of
magnitude (biomass diffusion ~1e-2, soil diffusion ~3e1) while Adam's per-parameter
normalisation makes its step size scale-free in *update* terms but not in
*relative* terms: a step that is small for a coefficient near 30 is enormous for
one near 0.01. Defaults to `false`, which reproduces the Python loop exactly.
"""
Base.@kwdef struct TrainConfig
    epochs::Int = 7500
    learning_rate::Float64 = 0.2
    beta1::Float64 = 0.9
    beta2::Float64 = 0.95
    eps::Float64 = 1e-8
    lr_decay::Float64 = 0.999
    grad_clip::Union{Nothing,Float64} = 10.0
    param_min::Float64 = 1e-4
    param_max::Float64 = 1e4
    save_interval::Int = 10
    print_interval::Int = 100
    verbose::Bool = true
    chunk_size::Int = DEFAULT_CHUNK
    logspace::Bool = false
end

"""
    DEFAULT_CHUNK

Number of derivative components carried per forward-mode pass.

Taking all nine at once looks optimal on paper — one pass instead of two — but is
about 2.3x slower in practice on a 131x140 grid, because a `Dual{9,Float64}` field
is 80 bytes per cell and the working set falls out of L2. Splitting into two
narrower passes keeps the state resident and wins despite doing more total work.

Five is the measured optimum for grids of this size. It is worth re-timing with
[`autotune_chunk`](@ref) for a substantially different grid.
"""
const DEFAULT_CHUNK = 5

"""Hyper-parameters used by `train_invPDE_synthetic_batch.py`."""
const SYNTHETIC_TRAIN_CONFIG = TrainConfig(
    epochs = 10_000,
    learning_rate = 0.1,
    lr_decay = 0.9999,
    grad_clip = nothing,
    print_interval = 10,
)

"""
    TrainResult

Outcome of a fit: the fitted parameters, the loss curve, periodic parameter
snapshots (one every `save_interval` epochs, matching the Python
`parameter_history`), and wall-clock time.

`converged` is false when the run was cut short by a non-finite loss or gradient.
Note that `final_loss` then holds the last *finite* loss rather than `NaN` — the
Python synthetic script records the `NaN` instead. Filter on `converged` rather
than on `isfinite(final_loss)`; a diverged run can have both a finite loss and
useless parameters.
"""
struct TrainResult
    params::RietkerkParams{Float64}
    initial_params::RietkerkParams{Float64}
    loss_history::Vector{Float64}
    parameter_history::Vector{Dict{String,Float64}}
    epochs_run::Int
    final_loss::Float64
    elapsed_seconds::Float64
    converged::Bool
    seed::Union{Nothing,Int}
    """
    Gradient norm at each epoch, in the space being optimised.

    Recorded because the loss alone cannot distinguish a fit that has converged
    from one that stopped where the schedule ran out — the distinction the
    identifiability analysis rests on. Under `logspace` these are norms of dL/du,
    which are not comparable to raw-space norms; compare like with like.

    Defaults to empty so the eight-field positional constructor keeps working.
    """
    gradnorm_history::Vector{Float64}
end

TrainResult(p, ip, lh, ph, er, fl, es, c, s) =
    TrainResult(p, ip, lh, ph, er, fl, es, c, s, Float64[])

function Base.show(io::IO, r::TrainResult)
    print(io, "TrainResult(final_loss = ", @sprintf("%.6g", r.final_loss),
          ", epochs = ", r.epochs_run,
          ", ", @sprintf("%.1f", r.elapsed_seconds), "s",
          r.converged ? "" : ", DIVERGED", ")")
end

"""
    train(prob, θ0; cfg = TrainConfig(), seed = nothing) -> TrainResult

Fit the nine ecological coefficients by gradient descent through the simulation.

Gradients come from forward-mode AD (`ForwardDiff`) over the whole rollout. With
only nine parameters this is the right trade: one dual-number pass costs about ten
primal passes but allocates nothing beyond the state, whereas reverse mode would
have to tape thousands of Euler steps. Swapping in a reverse-mode backend means
replacing the `ForwardDiff.gradient!` call below and nothing else.

Each epoch: evaluate loss and gradient, optionally clip the gradient norm, take an
Adam step, decay the learning rate, then clamp parameters into
`[param_min, param_max]`.

`lossfn(θ, prob)` selects the objective. The default [`loss`](@ref) uses the
fixed-step backend. Pass a closure over [`ode_loss`](@ref) to fit against the
adaptive ODE backend instead — roughly 10–100× slower depending on solver and
tolerance, so budget accordingly:

```julia
train(prob, θ0; lossfn = (v, pr) -> ode_loss(v, pr; alg = Tsit5(),
                                             abstol = 1e-2, reltol = 1e-4))
```

`callback(epoch, θ, loss, gradnorm)` is invoked every `save_interval` epochs, right
after the parameter snapshot is taken, with `θ` in natural units regardless of
`logspace` and `gradnorm` the *unclipped* norm in the optimised space. It exists so
a long fit can be observed while it runs — `TrainResult` only materialises at the
end, which on a multi-hour fit means hours with nothing to look at. Mutating `θ`
from the callback corrupts the fit; copy it if you need to keep it.
"""
function train(prob::InverseProblem, θ0::AbstractVector{<:Real};
               cfg::TrainConfig = TrainConfig(), seed::Union{Nothing,Integer} = nothing,
               lossfn = loss, callback = nothing)
    θ = collect(Float64, θ0)
    length(θ) == NPARAMS || throw(DimensionMismatch("expected $NPARAMS parameters"))
    initial = RietkerkParams(copy(θ))

    # `x` is what Adam moves; `θ` is what the model sees. In log-space the
    # objective closes over `exp`, so ForwardDiff returns dL/du directly and the
    # chain rule never appears by hand. The clamp is applied in x-space to the
    # image of [param_min, param_max], which is the same feasible set.
    lo = cfg.logspace ? log(cfg.param_min) : cfg.param_min
    hi = cfg.logspace ? log(cfg.param_max) : cfg.param_max
    x = cfg.logspace ? log.(clamp.(θ, cfg.param_min, cfg.param_max)) : θ

    objective = let prob = prob, lossfn = lossfn, logspace = cfg.logspace
        logspace ? (v -> lossfn(exp.(v), prob)) : (v -> lossfn(v, prob))
    end
    result_buffer = DiffResults.GradientResult(x)
    gradcfg = ForwardDiff.GradientConfig(objective, x, ForwardDiff.Chunk{cfg.chunk_size}())

    opt = AdamState(NPARAMS; lr = cfg.learning_rate, beta1 = cfg.beta1,
                    beta2 = cfg.beta2, eps = cfg.eps)

    loss_history = Float64[]
    gradnorm_history = Float64[]
    parameter_history = Dict{String,Float64}[]
    converged = true
    t0 = time()

    for epoch in 0:(cfg.epochs - 1)
        result_buffer = ForwardDiff.gradient!(result_buffer, objective, x, gradcfg)
        lossval = DiffResults.value(result_buffer)
        grad = DiffResults.gradient(result_buffer)

        if !isfinite(lossval)
            cfg.verbose && @warn "non-finite loss; stopping" epoch loss=lossval
            converged = false
            break
        end
        if any(!isfinite, grad)
            cfg.verbose && @warn "non-finite gradient; stopping" epoch
            converged = false
            push!(loss_history, lossval)
            break
        end

        push!(loss_history, lossval)
        # Before clipping: the clipped norm would just read back as grad_clip.
        push!(gradnorm_history, sqrt(sum(abs2, grad)))

        cfg.grad_clip === nothing || clip_global_norm!(grad, cfg.grad_clip)
        adam_step!(opt, x, grad)
        decay_lr!(opt, cfg.lr_decay)
        clamp!(x, lo, hi)
        cfg.logspace && (θ .= exp.(x))

        # Snapshot *after* the step, matching the Python `parameter_history`:
        # the entry labelled epoch 0 holds the parameters that epoch 0 produced,
        # not the ones it started from. Those are in `initial_params`.
        if epoch % cfg.save_interval == 0
            push!(parameter_history, _snapshot(epoch, θ))
            # Same cadence for the callback, so a caller streaming to disk writes
            # exactly the rows `parameter_history` will hold. Anything watching a
            # multi-hour fit needs this: the result object only exists at the end.
            callback === nothing ||
                callback(epoch, θ, lossval, gradnorm_history[end])
        end

        if cfg.verbose && (epoch % cfg.print_interval == 0 || epoch == cfg.epochs - 1)
            @printf("Epoch %5d  loss %.6g  lr %.4g\n", epoch, lossval, opt.lr)
            flush(stdout)
        end
    end

    last_epoch = length(loss_history) - 1
    if last_epoch >= 0 && (isempty(parameter_history) ||
                           parameter_history[end]["epoch"] != last_epoch)
        push!(parameter_history, _snapshot(last_epoch, θ))
    end

    return TrainResult(RietkerkParams(copy(θ)), initial, loss_history, parameter_history,
                       length(loss_history),
                       isempty(loss_history) ? NaN : loss_history[end],
                       time() - t0, converged,
                       seed === nothing ? nothing : Int(seed),
                       gradnorm_history)
end

"""
    train(prob; cfg, seed) -> TrainResult

Fit from a random start. `seed` makes the initialisation reproducible and is
recorded in the result.
"""
function train(prob::InverseProblem; cfg::TrainConfig = TrainConfig(),
               seed::Union{Nothing,Integer} = nothing, lossfn = loss)
    rng = seed === nothing ? Random.default_rng() : Random.Xoshiro(seed)
    return train(prob, paramvector(randparams(rng)); cfg = cfg, seed = seed, lossfn = lossfn)
end

"""
    train_many(prob, seeds; cfg = TrainConfig(), parallel = :runs) -> Vector{TrainResult}

Fit the model from several random restarts. The published workflow trains 30–100
models and then filters and pools them, because the objective is non-convex and any
single fit can land on a degenerate solution.

`parallel` picks where the threads go:

- `:runs` (default) — one thread per restart, each fit running serially. Best when
  there are more restarts than cores, which is the usual case.
- `:sites` — restarts run one after another, each using threads across sites
  internally. Best for a single long fit, or when memory is tight.
- `:none` — fully serial; use for reproducible timing.

Results come back in `seeds` order regardless.
"""
function train_many(prob::InverseProblem, seeds::AbstractVector{<:Integer};
                    cfg::TrainConfig = TrainConfig(), parallel::Symbol = :runs, lossfn = loss)
    parallel in (:runs, :sites, :none) ||
        throw(ArgumentError("parallel must be :runs, :sites or :none, got :$parallel"))
    results = Vector{TrainResult}(undef, length(seeds))

    if parallel === :runs && Threads.nthreads() > 1
        # Each run needs the site loop serial, or the two levels of threading fight.
        inner = rethread(prob, false)
        quiet = TrainConfig(cfg; verbose = false)
        Threads.@threads for i in eachindex(seeds)
            results[i] = train(inner; cfg = quiet, seed = seeds[i], lossfn = lossfn)
        end
    else
        outer = parallel === :sites ? rethread(prob, true) : prob
        for i in eachindex(seeds)
            cfg.verbose && @info "fitting run $i/$(length(seeds))" seed = seeds[i]
            results[i] = train(outer; cfg = cfg, seed = seeds[i], lossfn = lossfn)
        end
    end
    return results
end

"""
    TrainConfig(base; kwargs...)

Copy `base` with selected fields replaced.
"""
function TrainConfig(base::TrainConfig; kwargs...)
    fields = (; (f => getfield(base, f) for f in fieldnames(TrainConfig))...)
    return TrainConfig(; merge(fields, NamedTuple(kwargs))...)
end

function _snapshot(epoch::Integer, θ::AbstractVector)
    d = Dict{String,Float64}("epoch" => Float64(epoch))
    for i in 1:NPARAMS
        d[PARAM_NAMES[i]] = θ[i]
    end
    return d
end

"""
    loss_and_gradient(prob, θ; chunk_size = DEFAULT_CHUNK) -> (loss, gradient)

Single loss/gradient evaluation, exposed for diagnostics and gradient checks.
"""
function loss_and_gradient(prob::InverseProblem, θ::AbstractVector{<:Real};
                           chunk_size::Integer = DEFAULT_CHUNK)
    v = collect(Float64, θ)
    objective = let prob = prob
        x -> loss(x, prob)
    end
    res = DiffResults.GradientResult(v)
    gradcfg = ForwardDiff.GradientConfig(objective, v, ForwardDiff.Chunk{chunk_size}())
    res = ForwardDiff.gradient!(res, objective, v, gradcfg)
    return DiffResults.value(res), DiffResults.gradient(res)
end

"""
    autotune_chunk(prob, θ; candidates = (2, 3, 5, 9)) -> (best, timings)

Time one gradient evaluation at each candidate chunk width and return the fastest,
along with the full timing table. Worth running once for a new grid size — the
optimum is set by cache capacity, not by parameter count.
"""
function autotune_chunk(prob::InverseProblem, θ::AbstractVector{<:Real};
                        candidates = (2, 3, 5, 9))
    timings = Dict{Int,Float64}()
    for c in candidates
        loss_and_gradient(prob, θ; chunk_size = c)   # warm up / compile
        t0 = time()
        loss_and_gradient(prob, θ; chunk_size = c)
        timings[c] = time() - t0
    end
    best = argmin(timings)
    return (best, timings)
end
