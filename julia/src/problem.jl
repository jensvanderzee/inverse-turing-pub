"""
    SiteTrajectory{T}

One site's contribution to the inverse problem, in the form the loss needs:

- `initial_biomass`: field the simulation is launched from.
- `initial_target`: observed field the first predicted change is measured against.
  Usually equal to `initial_biomass`, but kept separate because the synthetic and
  real-data setups arrive at it differently.
- `forcings[k]`: weekly precipitation rates driving transition `k`.
- `targets[k]`: observed biomass field at the end of transition `k`.

The synthetic and real-data experiments differ only in how these fields are
filled, which is why they share a single loss function.
"""
struct SiteTrajectory{T,F<:AbstractVector}
    initial_biomass::Matrix{T}
    initial_target::Matrix{T}
    forcings::Vector{F}
    targets::Vector{Matrix{T}}

    function SiteTrajectory(initial_biomass::Matrix{T}, initial_target::Matrix{T},
                            forcings::Vector{F}, targets::Vector{Matrix{T}}) where {T,F<:AbstractVector}
        length(forcings) == length(targets) ||
            throw(DimensionMismatch("need one forcing per target, got $(length(forcings)) and $(length(targets))"))
        for t in targets
            size(t) == size(initial_biomass) ||
                throw(DimensionMismatch("target size $(size(t)) != initial biomass size $(size(initial_biomass))"))
        end
        new{T,F}(initial_biomass, initial_target, forcings, targets)
    end
end

ntransitions(tr::SiteTrajectory) = length(tr.targets)

"""
    SiteTrajectory(series::SiteSeries)

Build a trajectory from consecutive observations of a real site: transition `k`
runs from year `k` to year `k+1` under year `k`'s precipitation.
"""
function SiteTrajectory(series::SiteSeries{T}) where {T}
    obs = series.observations
    length(obs) >= 2 ||
        throw(ArgumentError("site $(series.name) has $(length(obs)) observations; need at least 2"))
    initial = copy(obs[1].biomass)
    forcings = [o.weekly_precipitation for o in obs[1:(end - 1)]]
    targets = [copy(o.biomass) for o in obs[2:end]]
    return SiteTrajectory(initial, copy(obs[1].biomass), forcings, targets)
end

"""
    InverseProblem(trajectories, cfg; average = true, threaded = Threads.nthreads() > 1)

Everything the fit needs: the observed trajectories, the time discretisation, and
whether the per-transition MSEs are averaged or summed.

`average = true` matches `realdata_train_invPDE.py` (loss divided by the number of
comparisons); `average = false` matches `train_invPDE_synthetic_batch.py`, which
sums. The difference rescales the gradient and therefore the effective learning
rate, so it is not cosmetic.

`delta_loss = true` scores the year-on-year *change* in biomass; `false` scores the
absolute field. This is `use_delta_loss` in the Python code. Every published fit
uses the delta form — see [`mean_squared_delta_error`](@ref) for why — but the
absolute form is kept because it is what makes the difference visible.

`threaded` evaluates sites in parallel. Sites are independent, so this is the only
parallelism a single loss evaluation offers — the transitions within a site are
inherently sequential. Partial sums are reduced in a fixed order, so results do not
depend on the thread count. Leave it off when the *outer* loop is already parallel
(fitting many random restarts at once), which scales further.
"""
struct InverseProblem{T,F}
    trajectories::Vector{SiteTrajectory{T,F}}
    cfg::SimConfig
    average::Bool
    threaded::Bool
    delta_loss::Bool
end

InverseProblem(trajectories::Vector{<:SiteTrajectory}, cfg::SimConfig;
               average::Bool = true, threaded::Bool = Threads.nthreads() > 1,
               delta_loss::Bool = true) =
    InverseProblem(trajectories, cfg, average, threaded, delta_loss)

"""
    InverseProblem(sites::AbstractVector{<:SiteSeries}, cfg; average = true, ...)

Convenience constructor for the real-data fit.
"""
InverseProblem(sites::AbstractVector{<:SiteSeries}, cfg::SimConfig;
               average::Bool = true, threaded::Bool = Threads.nthreads() > 1,
               delta_loss::Bool = true) =
    InverseProblem([SiteTrajectory(s) for s in sites], cfg, average, threaded, delta_loss)

"""
    rethread(prob, threaded) -> InverseProblem

Copy of `prob` with the site-level threading flag flipped. Used by
[`train_many`](@ref) to keep the inner loop serial when the outer loop is threaded.
"""
rethread(prob::InverseProblem, threaded::Bool) =
    InverseProblem(prob.trajectories, prob.cfg, prob.average, threaded, prob.delta_loss)

ntransitions(prob::InverseProblem) = sum(ntransitions, prob.trajectories; init = 0)
Base.length(prob::InverseProblem) = length(prob.trajectories)

function Base.show(io::IO, prob::InverseProblem{T}) where {T}
    print(io, "InverseProblem{$T}(", length(prob.trajectories), " sites, ",
          ntransitions(prob), " transitions, ",
          prob.cfg.steps_per_week, " steps/week, ",
          prob.average ? "mean" : "sum", " ",
          prob.delta_loss ? "delta" : "absolute", " loss)")
end

"""
    mean_squared_delta_error(pred, prev_pred, target, prev_target)

Mean squared error between the *change* the model predicts over a transition and
the *change* actually observed.

Fitting on differences rather than absolute fields removes the site's static
spatial mean from the objective, so the parameters are constrained by how the
vegetation moves rather than by how much of it there is. This is `use_delta_loss`
in the Python code, and it is the loss all published fits use.
"""
function mean_squared_delta_error(pred::AbstractMatrix, prev_pred::AbstractMatrix,
                                  target::AbstractMatrix, prev_target::AbstractMatrix)
    acc = zero(promote_type(eltype(pred), eltype(target)))
    @inbounds @simd for i in eachindex(pred, prev_pred, target, prev_target)
        d = (pred[i] - prev_pred[i]) - (target[i] - prev_target[i])
        acc += d * d
    end
    return acc / length(pred)
end

"""
    mean_squared_error(pred, target)

Plain MSE on absolute fields — the `use_delta_loss = false` branch.
"""
function mean_squared_error(pred::AbstractMatrix, target::AbstractMatrix)
    acc = zero(promote_type(eltype(pred), eltype(target)))
    @inbounds @simd for i in eachindex(pred, target)
        d = pred[i] - target[i]
        acc += d * d
    end
    return acc / length(pred)
end

"""
    loss(θ, prob) -> Real

Objective for the inverse problem: roll every site forward under its observed
forcing and accumulate the per-transition MSE against the observations — on
year-on-year differences by default, on absolute fields when
`prob.delta_loss = false`.

`θ` is the flat 9-vector in [`PARAM_NAMES`](@ref) order. The state is allocated at
`eltype(θ)`, so passing a vector of `ForwardDiff.Dual`s differentiates the whole
rollout — this is the function [`train`](@ref) hands to ForwardDiff.
"""
function loss(θ::AbstractVector, prob::InverseProblem)
    p = RietkerkParams(θ)
    T = eltype(θ)
    trajectories = prob.trajectories
    partials = Vector{T}(undef, length(trajectories))

    if prob.threaded && length(trajectories) > 1
        Threads.@threads for i in eachindex(trajectories)
            partials[i] = trajectory_loss(trajectories[i], p, prob.cfg, T; delta = prob.delta_loss)
        end
    else
        for i in eachindex(trajectories)
            partials[i] = trajectory_loss(trajectories[i], p, prob.cfg, T; delta = prob.delta_loss)
        end
    end

    total = sum(partials)  # fixed order, so the result is thread-count independent
    n = ntransitions(prob)
    return prob.average && n > 0 ? total / n : total
end

"""
    trajectory_loss(tr, p, cfg, T; delta = true) -> T

Summed MSE over one site's transitions. Split out from [`loss`](@ref) so the site
loop can be threaded.
"""
function trajectory_loss(tr::SiteTrajectory, p::RietkerkParams, cfg::SimConfig,
                         ::Type{T}; delta::Bool = true) where {T}
    state = simstate(tr.initial_biomass; T = T)
    prev_pred = copy(state.biomass)
    prev_target = tr.initial_target
    acc = zero(T)
    for k in eachindex(tr.targets)
        simulate_year!(state, p, tr.forcings[k], cfg)
        target = tr.targets[k]
        acc += delta ?
               mean_squared_delta_error(state.biomass, prev_pred, target, prev_target) :
               mean_squared_error(state.biomass, target)
        copyto!(prev_pred, state.biomass)
        prev_target = target
    end
    return acc
end

loss(p::RietkerkParams, prob::InverseProblem) = loss(paramvector(p), prob)

"""
    evaluate(p, series, cfg) -> NamedTuple

Forward-only diagnostics for one site: mean delta-MSE, mean delta-MAE, and the
mean pixelwise Pearson correlation between the predicted and observed biomass
field at each step. Mirrors `evaluate_model_on_site` in `realdata_test_invPDE.py`,
and is what model selection on held-out sites is based on.
"""
function evaluate(p::RietkerkParams, series::SiteSeries{T}, cfg::SimConfig) where {T}
    obs = series.observations
    length(obs) >= 2 || return (mse = NaN, mae = NaN, correlation = NaN, num_transitions = 0)

    state = simstate(obs[1].biomass)
    total_mse = 0.0
    total_mae = 0.0
    corrs = Float64[]

    for k in 1:(length(obs) - 1)
        observed_delta = obs[k + 1].biomass .- obs[k].biomass
        prev_pred = copy(state.biomass)
        simulate_year!(state, p, obs[k].weekly_precipitation, cfg)

        mse = 0.0
        mae = 0.0
        @inbounds for i in eachindex(state.biomass)
            d = (state.biomass[i] - prev_pred[i]) - observed_delta[i]
            mse += d * d
            mae += abs(d)
        end
        n = length(state.biomass)
        total_mse += mse / n
        total_mae += mae / n

        push!(corrs, _pearson(state.biomass, obs[k + 1].biomass))
    end

    nt = length(obs) - 1
    return (mse = total_mse / nt, mae = total_mae / nt,
            correlation = Statistics.mean(corrs), num_transitions = nt)
end

"""Pearson correlation over all pixels; zero when either field is constant."""
function _pearson(a::AbstractArray, b::AbstractArray)
    sa = Statistics.std(vec(a); corrected = false)
    sb = Statistics.std(vec(b); corrected = false)
    (sa > 0 && sb > 0) || return 0.0
    return Float64(Statistics.cor(vec(Float64.(a)), vec(Float64.(b))))
end
