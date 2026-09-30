"""
    basefloat(x)

The underlying floating-point type of a value, seeing through `ForwardDiff.Dual`
nesting. Used to round `dt` and precipitation rates to the working precision
*before* they enter the update, which is what PyTorch does when a Python `float`
meets a `float32` tensor.
"""
basefloat(::Type{T}) where {T<:AbstractFloat} = T
basefloat(::Type{<:ForwardDiff.Dual{<:Any,V,<:Any}}) where {V} = basefloat(V)
basefloat(x) = basefloat(typeof(x))

"""
    SimConfig(; steps_per_week, year_time_units = 1.0)

Time discretisation of one simulated year.

- `steps_per_week`: number of explicit Euler sub-steps per forcing week.
- `year_time_units`: how much model time one calendar year spans. The real-data
  scripts use `1.0`; `train_invPDE_synthetic_batch.py` uses `1.5`. This single
  number is the only difference between the two integrators in the Python code,
  and it changes the fitted rate constants, so it is made explicit here rather
  than hard-coded.

The step size is `year_time_units / (num_weeks * steps_per_week)`.
"""
Base.@kwdef struct SimConfig
    steps_per_week::Int
    year_time_units::Float64 = 1.0
end

"""
    SimState(surface_water, soil_water, biomass)

Mutable simulation state: the three coupled fields plus one scratch buffer that
the Laplacian writes into. All four arrays share element type and shape.

Construct from an initial biomass field with [`simstate`](@ref).
"""
struct SimState{T,A<:AbstractMatrix{T}}
    surface_water::A
    soil_water::A
    biomass::A
    scratch::A

    function SimState(surface_water::A, soil_water::A, biomass::A) where {T,A<:AbstractMatrix{T}}
        size(surface_water) == size(soil_water) == size(biomass) ||
            throw(DimensionMismatch("all three fields must have the same size"))
        new{T,A}(surface_water, soil_water, biomass, similar(biomass))
    end
end

"""
    simstate(biomass; T = eltype(biomass))

Build a [`SimState`](@ref) from an initial biomass field, with surface and soil
water starting at zero — the initialisation used everywhere in the original code.

Passing `T` promotes the state, which is how a `Float64` biomass field is lifted
to `ForwardDiff.Dual` for a gradient evaluation.
"""
function simstate(biomass::AbstractMatrix; T::Type = eltype(biomass))
    b = convert.(T, biomass)
    SimState(zero(b), zero(b), b)
end

Base.size(s::SimState) = size(s.biomass)
Base.eltype(::SimState{T}) where {T} = T

"""
    copystate(s)

Deep copy of a state, so a rollout can branch without disturbing the original.
"""
copystate(s::SimState) = SimState(copy(s.surface_water), copy(s.soil_water), copy(s.biomass))

"""
    step!(state, p, precip_rate, dt) -> state

One explicit Euler step of the Rietkerk-type system

    ∂ₜO = d₁ ∇²O − l₁ O + R − r₂ O B
    ∂ₜW = d₂ ∇²W − l₂ W + r₂ O B − r₁ W B
    ∂ₜB = d₃ ∇²B − l₃ B + j r₁ W B

for surface water `O`, soil water `W` and biomass `B`.

!!! important "Update order is part of the discretisation"
    The three fields are advanced **sequentially, not simultaneously**: the soil
    water equation uses the *already updated* surface water, and the biomass
    equation uses the *already updated* soil water. This is a Gauss–Seidel-style
    sweep rather than a true forward Euler step, and it is what the PyTorch code
    does — each Python assignment rebinds the name before the next expression is
    built. It is almost certainly incidental rather than intended.

    Consequently this is *not* a method-of-lines discretisation of an ODE, and
    handing the right-hand side to an ODE solver would give the simultaneous
    variant instead.

    How much it matters, measured on the published best-fit parameters at
    4 steps/week: the two orders differ by ~1e-6 relative in nine-year delta-MSE
    on well-behaved sites and ~5e-4 on the stiffest one (subsite_j). That is far
    below the scatter between random restarts, so the *science* does not hinge on
    it — but it is above the 1e-8 agreement the port achieves against the Python
    outputs, so reproducing those numbers does.
"""
function step!(s::SimState, p::RietkerkParams, precip_rate, dt)
    O, W, B, lap = s.surface_water, s.soil_water, s.biomass, s.scratch

    # Hoisting the coefficients out of the broadcast keeps `getfield` off the hot
    # path and stops `@.` from ever seeing a property access.
    d1 = p.surface_water_diffusion_coeff
    d2 = p.soil_water_diffusion_coeff
    d3 = p.biomass_diffusion_coeff
    l1 = p.evaporation_rate
    l2 = p.seepage_rate
    l3 = p.mortality_rate
    r2 = p.infiltration_rate
    r1 = p.plant_uptake_rate
    j = p.water_use_efficiency

    laplacian!(lap, O)
    @. O += (d1 * lap - l1 * O + precip_rate - r2 * O * B) * dt

    laplacian!(lap, W)
    @. W += (d2 * lap - l2 * W + r2 * O * B - r1 * W * B) * dt

    laplacian!(lap, B)
    @. B += (d3 * lap - l3 * B + j * r1 * W * B) * dt

    return s
end

"""
    simulate_year!(state, p, weekly_precip, cfg) -> state

Advance the state by one year under a weekly precipitation forcing.

`weekly_precip[w]` is a *rate* in mm/day for week `w`. Each week delivers
`weekly_precip[w] * 7` mm, split evenly across `cfg.steps_per_week` sub-steps and
re-expressed as a rate so that `rate * dt` equals the intended volume per step.
Because `dt` shrinks as `steps_per_week` grows, the injected rate grows to match —
total annual water is invariant to the discretisation, but the peak forcing is not.
"""
function simulate_year!(s::SimState, p::RietkerkParams, weekly_precip::AbstractVector, cfg::SimConfig)
    nweeks = length(weekly_precip)
    nweeks > 0 || throw(ArgumentError("weekly_precip must be non-empty"))
    cfg.steps_per_week > 0 || throw(ArgumentError("steps_per_week must be positive"))

    F = basefloat(eltype(s.biomass))
    total_steps = nweeks * cfg.steps_per_week
    dt = F(cfg.year_time_units / total_steps)

    @inbounds for w in 1:nweeks
        volume_per_step = weekly_precip[w] * 7 / cfg.steps_per_week
        rate = F(volume_per_step / dt)
        for _ in 1:cfg.steps_per_week
            step!(s, p, rate, dt)
        end
    end
    return s
end

"""
    diffusion_stability_limit(cfg; num_weeks = 52) -> Float64

Largest diffusion coefficient the explicit scheme can integrate without blowing up,
for the given time discretisation.

Von Neumann analysis of `uⁿ⁺¹ = uⁿ + dt·d·∇²uⁿ` with the 5-point stencil gives an
amplification factor of `1 + d·dt·(2cos kx + 2cos ky − 4)`, worst at `kx = ky = π`,
so stability needs `d·dt ≤ 1/4`. With `dt = year_time_units / (num_weeks ·
steps_per_week)` the bound is

    d ≤ num_weeks · steps_per_week / (4 · year_time_units)

The coefficients fitted to the real data land around 18–34, against limits of 26 at
`steps_per_week = 2`, 39 at 3 and 52 at 4 — so this bound does bind at coarse
settings.

!!! warning "Necessary but not sufficient — prefer [`spectral_bound`](@ref)"
    This accounts for diffusion only. The reaction terms `r₂B` and `r₁B` impose
    their own limit, and with biomass running to the NDVI ceiling of 1500 g/m²
    they are frequently the larger constraint. Audited against the 47 published
    fits, this bound flags **none** of the 15 parameter sets that are actually
    under-resolved at the setting they were fitted with; those have *small*
    diffusion coefficients and large reaction rates.

    Use [`stability_ratio`](@ref)/[`check_stability`](@ref), which combine both.
    This function is retained because it is the exact criterion for the pure
    diffusion sub-problem and is what sets the ceiling during a fit, when the
    optimiser can push diffusion coefficients far above their fitted values.
"""
diffusion_stability_limit(cfg::SimConfig; num_weeks::Integer = 52) =
    num_weeks * cfg.steps_per_week / (4 * cfg.year_time_units)

"""
    spectral_bound(p, biomass_max) -> Float64

Estimate of the largest decay rate in the linearised system — the quantity that
sets the explicit step-size limit.

Per equation, the fastest linear decay is

| field        | rate                          |
|:-------------|:------------------------------|
| surface water| `8 d₁ + l₁ + r₂ B`            |
| soil water   | `8 d₂ + l₂ + r₁ B`            |
| biomass      | `8 d₃ + l₃`                   |

where `8 d` is the extreme eigenvalue of the 5-point Laplacian and `B` is the
largest biomass in play. The bound is the maximum of the three.

!!! warning "Diffusion is often *not* the binding term"
    [`diffusion_stability_limit`](@ref) accounts only for `8 d`. On the published
    real-data fits the reaction terms `r₂B` and `r₁B` are comparable or larger,
    because biomass runs to the NDVI ceiling of 1500 g/m². Across the 47 published
    parameter sets the diffusion-only bound flags **none** of the 15 that are
    actually inaccurate at the production setting. Prefer this function.
"""
function spectral_bound(p::RietkerkParams, biomass_max::Real)
    B = Float64(biomass_max)
    return max(8 * p.surface_water_diffusion_coeff + p.evaporation_rate + p.infiltration_rate * B,
               8 * p.soil_water_diffusion_coeff + p.seepage_rate + p.plant_uptake_rate * B,
               8 * p.biomass_diffusion_coeff + p.mortality_rate)
end

"""
    max_stable_dt(p, biomass_max) -> Float64

Largest explicit-Euler step the parameters admit, `2 / spectral_bound(p, B)`.
"""
max_stable_dt(p::RietkerkParams, biomass_max::Real) = 2 / spectral_bound(p, biomass_max)

"""
    stability_ratio(p, cfg, biomass_max; num_weeks = 52) -> Float64

`dt / max_stable_dt` — how far the chosen discretisation sits past the linearised
explicit limit. Larger is worse.

Calibration against the 47 published real-data fits, integrated for one year at
`steps_per_week = 3` with `biomass_max = 1500` and scored against an adaptive
`ROCK2` reference:

| ratio       | outcome                                        |
|:------------|:-----------------------------------------------|
| 1.26 – 1.45 | all 32 accurate (relative error ≤ 1%)          |
| 2.72 – 5.11 | all 15 inaccurate (> 1%) or divergent to `NaN` |

The two groups separate cleanly with nothing in between, so **2 is the practical
threshold**. Values above 1 are tolerable because the estimate takes a max over
equations rather than the true coupled Jacobian spectrum, and because `biomass_max`
is the NDVI ceiling rather than a typical value — both make it conservative. Treat
it as a calibrated diagnostic, not an exact criterion.

Run `scripts/accuracy_audit.jl` to reproduce the calibration on your own fits.
"""
function stability_ratio(p::RietkerkParams, cfg::SimConfig, biomass_max::Real;
                         num_weeks::Integer = 52)
    dt = cfg.year_time_units / (num_weeks * cfg.steps_per_week)
    return dt / max_stable_dt(p, biomass_max)
end

"""
    check_stability(p, cfg, biomass_max; num_weeks = 52, threshold = 2.0, warn = true) -> Bool

Whether the discretisation is safe for these parameters, using the combined
diffusion-plus-reaction bound. Warns with the offending ratio when it is not,
which is the usual explanation for a rollout that turns to `NaN` or for a fit that
disagrees with an adaptive solve.

`biomass_max` should be the largest biomass the rollout will see — for NDVI-derived
data that is the `ndvi_to_biomass_multiplier` ceiling, typically 1500.
"""
function check_stability(p::RietkerkParams, cfg::SimConfig, biomass_max::Real;
                         num_weeks::Integer = 52, threshold::Real = 2.0, warn::Bool = true)
    ratio = stability_ratio(p, cfg, biomass_max; num_weeks = num_weeks)
    ok = ratio <= threshold
    if !ok && warn
        @warn """explicit step is past the linearised stability limit; expect large
                 discretisation error or divergence. Raise steps_per_week — the
                 ratio scales as 1/steps_per_week.""" ratio threshold steps_per_week=cfg.steps_per_week needed_steps_per_week=ceil(Int, cfg.steps_per_week * ratio / threshold)
    end
    return ok
end

"""
    simulate_years!(state, p, weekly_precip, cfg, nyears; callback = nothing)

Repeat [`simulate_year!`](@ref) `nyears` times under a fixed forcing profile.

`callback(year, state)` is invoked after each year, which is how the bifurcation
and extrapolation scripts record mean-biomass trajectories without allocating a
snapshot per year.
"""
function simulate_years!(s::SimState, p::RietkerkParams, weekly_precip::AbstractVector,
                         cfg::SimConfig, nyears::Integer; callback = nothing)
    for year in 1:nyears
        simulate_year!(s, p, weekly_precip, cfg)
        callback === nothing || callback(year, s)
    end
    return s
end
