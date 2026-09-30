"""
    equilibrium_biomass(p, grid; precipitation, years, cfg, rng, T = Float64) -> Matrix{T}

Spin the forward model up to a spatially patterned steady state under constant
rainfall, and return the final biomass field.

All three fields start from `Uniform[0, 10)` noise; the patterning that emerges is
the Turing instability the whole study is about, so the result depends on `rng`.
Port of `SyntheticDataGenerator.generate_equilibrium_state`.
"""
function equilibrium_biomass(p::RietkerkParams, grid::Tuple{Int,Int};
                             precipitation::Real, years::Integer, cfg::SimConfig,
                             rng::Random.AbstractRNG = Random.default_rng(),
                             T::Type = Float64)
    H, W = grid
    state = SimState(T.(rand(rng, H, W) .* 10),
                     T.(rand(rng, H, W) .* 10),
                     T.(rand(rng, H, W) .* 10))
    weekly = sinusoidal_weekly_precip(precipitation; amplitude_fraction = 0.0)
    simulate_years!(state, p, weekly, cfg, years)
    return copy(state.biomass)
end

"""
    synthetic_series(p, equilibrium, weekly_precip; years, noise_level, cfg, rng) -> Vector{Matrix}

Generate one site's observed time series: run the ground-truth model forward from
`equilibrium` for `years` years under `weekly_precip`, recording the biomass field
after each year with additive Gaussian noise of standard deviation `noise_level`.

Noise is added to the *recorded* field only; the trajectory itself stays clean, as
in `SyntheticDataGenerator.generate_training_data`.
"""
function synthetic_series(p::RietkerkParams, equilibrium::AbstractMatrix,
                          weekly_precip::AbstractVector;
                          years::Integer, noise_level::Real, cfg::SimConfig,
                          rng::Random.AbstractRNG = Random.default_rng())
    T = eltype(equilibrium)
    state = simstate(equilibrium)
    out = Matrix{T}[]
    for _ in 1:years
        simulate_year!(state, p, weekly_precip, cfg)
        push!(out, state.biomass .+ T(noise_level) .* randn(rng, T, size(state.biomass)))
    end
    return out
end

"""
    synthetic_experiment(; p = SYNTHETIC_TRUTH, grid = (128, 128), n_sites = 4,
                         rain_multiplier = 0.75, annual_range = (15.0, 27.0),
                         equilibrium_precip = 21.0, equilibrium_years = 100,
                         years = 10, noise_level = 0.05, cfg, rng, T = Float64)

Build the full synthetic benchmark used by `train_invPDE_synthetic_batch.py`:
spin up one shared equilibrium field, then roll out `n_sites` sites whose annual
rainfall totals are spread evenly over `annual_range .* rain_multiplier`.

Returns `(problem, equilibrium, profiles, targets)` where `problem` is an
[`InverseProblem`](@ref) ready to hand to [`train`](@ref).

Note that `cfg` should carry `year_time_units = 1.5` to match the Python
synthetic script — see [`SimConfig`](@ref).
"""
function synthetic_experiment(; p::RietkerkParams = SYNTHETIC_TRUTH,
                              grid::Tuple{Int,Int} = (128, 128),
                              n_sites::Integer = 4,
                              rain_multiplier::Real = 0.75,
                              annual_range::Tuple{<:Real,<:Real} = (15.0, 27.0),
                              equilibrium_precip::Real = 21.0,
                              equilibrium_years::Integer = 100,
                              years::Integer = 10,
                              noise_level::Real = 0.05,
                              peak_week::Real = 26.0,
                              amplitude_fraction::Real = 0.7,
                              cfg::SimConfig = SimConfig(steps_per_week = 1, year_time_units = 1.5),
                              rng::Random.AbstractRNG = Random.default_rng(),
                              T::Type = Float64)
    equilibrium = equilibrium_biomass(p, grid;
                                      precipitation = equilibrium_precip * rain_multiplier,
                                      years = equilibrium_years, cfg = cfg, rng = rng, T = T)

    lo, hi = annual_range
    totals = n_sites == 1 ? [lo * rain_multiplier] :
             collect(range(lo * rain_multiplier, hi * rain_multiplier; length = n_sites))
    profiles = [sinusoidal_weekly_precip(t; peak_week = peak_week,
                                         amplitude_fraction = amplitude_fraction)
                for t in totals]

    targets = [synthetic_series(p, equilibrium, prof;
                                years = years, noise_level = noise_level,
                                cfg = cfg, rng = rng)
               for prof in profiles]

    trajectories = [SiteTrajectory(equilibrium, equilibrium, fill(prof, years), tgt)
                    for (prof, tgt) in zip(profiles, targets)]

    # The synthetic script sums the per-transition MSEs rather than averaging.
    problem = InverseProblem(trajectories, cfg; average = false)
    return (problem = problem, equilibrium = equilibrium,
            profiles = profiles, targets = targets, annual_totals = totals)
end
