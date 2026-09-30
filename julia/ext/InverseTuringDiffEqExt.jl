"""
    InverseTuringDiffEqExt

Adaptive ODE backend for InverseTuring, activated by
`using OrdinaryDiffEqStabilizedRK`.

That sub-package rather than the `OrdinaryDiffEq` meta-package: it carries the
`ROCK2`/`ROCK4` stabilised explicit methods this problem wants, plus `ODEProblem`
and `solve`, and loads in a fraction of the time. Other algorithm families work
too — add `using OrdinaryDiffEqTsit5` or `OrdinaryDiffEqSDIRK` and pass the `alg`
keyword.

This solves the *continuous* Rietkerk system by method of lines, in contrast to the
fixed-step Gauss–Seidel sweep the package uses by default. The two are different
discretisations of the same PDE, and the fitted parameters are not interchangeable
between them — see `scripts/diffeq_comparison.jl`.

The motivation is stiffness. The default explicit scheme is bounded by
`diffusion_stability_limit(cfg) = 52·steps_per_week/4`, and the coefficients fitted
to the real data (18–34) sit right against it. A stabilised explicit method such as
`ROCK2` has an extended real-axis stability region and removes that ceiling without
forming a Jacobian, which matters at ~55 000 unknowns.
"""
module InverseTuringDiffEqExt

using InverseTuring
using InverseTuring: WeeklyForcing, week_boundaries, rietkerk_rhs!, pack_state,
                     unpack_state, biomass_of, RietkerkParams, SimConfig,
                     SiteTrajectory, InverseProblem, mean_squared_delta_error,
                     mean_squared_error, NPARAMS
using OrdinaryDiffEqStabilizedRK
import Statistics

"""
    solve_year(p, u0, weekly_precip, cfg; alg = ROCK2(), save_everystep = false, kwargs...)

Integrate one year and return the `ODESolution`.

Week boundaries are passed as `tstops` because the forcing is piecewise constant;
without them an adaptive controller steps across the jumps and either drops order
or burns steps on rejections.

`save_everystep = false` keeps only the endpoints, which is all the loss needs and
avoids retaining a full trajectory of 55 000-element states.
"""
function InverseTuring.solve_year(p::RietkerkParams, u0::AbstractArray{<:Any,3},
                                  weekly_precip::AbstractVector, cfg::SimConfig;
                                  alg = ROCK2(), save_everystep::Bool = false,
                                  abstol = 1e-8, reltol = 1e-8, kwargs...)
    forcing = WeeklyForcing(weekly_precip, cfg.year_time_units)
    rhs! = let forcing = forcing
        (du, u, par, t) -> rietkerk_rhs!(du, u, par, forcing, t)
    end
    prob = ODEProblem{true}(rhs!, u0, (0.0, cfg.year_time_units), p)
    return solve(prob, alg; tstops = week_boundaries(forcing),
                 save_everystep = save_everystep, abstol = abstol, reltol = reltol,
                 kwargs...)
end

"""
    simulate_years_ode(p, biomass0, weekly_precip, cfg, nyears; kwargs...)
        -> (final_u, mean_biomass_per_year, total_steps)

Roll forward `nyears` years under a fixed annual forcing, restarting the solve each
year so the forcing stays a function on `[0, year_time_units]`.

Returns the final packed state, the mean biomass after each year (with the initial
value first, so the vector has `nyears + 1` entries), and the total number of
accepted solver steps — the honest measure of work done, since it is not fixed in
advance the way `steps_per_week` is.
"""
function InverseTuring.simulate_years_ode(p::RietkerkParams, biomass0::AbstractMatrix,
                                          weekly_precip::AbstractVector, cfg::SimConfig,
                                          nyears::Integer; kwargs...)
    u = pack_state(biomass0)
    means = Float64[Statistics.mean(biomass_of(u))]
    steps = 0
    for _ in 1:nyears
        sol = InverseTuring.solve_year(p, u, weekly_precip, cfg; kwargs...)
        u = sol.u[end]
        steps += length(sol.t) - 1
        push!(means, Statistics.mean(biomass_of(u)))
    end
    return (u, means, steps)
end

"""
    ode_loss(θ, prob; kwargs...)

The delta-MSE objective evaluated with the adaptive backend.

Differentiable through `ForwardDiff`: the solver is pure Julia and propagates dual
numbers, so no adjoint machinery is needed for nine parameters. Note that adaptive
step selection then depends on the dual-valued error estimate, which makes the
objective very slightly non-smooth in `θ` — tighten `abstol`/`reltol` if a gradient
check disagrees with finite differences.
"""
function InverseTuring.ode_loss(θ::AbstractVector, prob::InverseProblem; kwargs...)
    p = RietkerkParams(θ)
    T = eltype(θ)
    total = zero(T)
    n = 0
    for tr in prob.trajectories
        u = pack_state(tr.initial_biomass; T = T)
        prev_pred = copy(biomass_of(u))
        prev_target = tr.initial_target
        for k in eachindex(tr.targets)
            sol = InverseTuring.solve_year(p, u, tr.forcings[k], prob.cfg; kwargs...)
            u = sol.u[end]
            target = tr.targets[k]
            total += prob.delta_loss ?
                     mean_squared_delta_error(biomass_of(u), prev_pred, target, prev_target) :
                     mean_squared_error(biomass_of(u), target)
            prev_pred = copy(biomass_of(u))
            prev_target = target
            n += 1
        end
    end
    return prob.average && n > 0 ? total / n : total
end

end # module
