"""
    WeeklyForcing(rates, year_time_units)

Precipitation as a function of continuous time: piecewise constant, one level per
week, callable as `f(t)`.

`rates[w]` is the mm/day rate for week `w`, the same input the discrete integrator
takes. Over week `w` — a window of `year_time_units / nweeks` model-time units —
the returned rate delivers exactly `rates[w] * 7` mm, matching what
[`simulate_year!`](@ref) injects. Annual volume is therefore identical between the
two backends, which is what makes their trajectories comparable at all.

The result is discontinuous at week boundaries. Adaptive solvers must be given
those instants as `tstops`, or they will step across a jump and either lose order
or waste steps rejecting; the extension does this automatically.
"""
struct WeeklyForcing{V<:AbstractVector}
    rates::V
    year_time_units::Float64
end

@inline function (f::WeeklyForcing)(t)
    n = length(f.rates)
    week_len = f.year_time_units / n
    idx = clamp(floor(Int, t / week_len) + 1, 1, n)
    return @inbounds f.rates[idx] * 7 * n / f.year_time_units
end

"""
    week_boundaries(f::WeeklyForcing) -> Vector{Float64}

Times at which the forcing jumps. Pass as `tstops`.
"""
week_boundaries(f::WeeklyForcing) =
    collect(range(0.0, f.year_time_units; length = length(f.rates) + 1))

"""
    rietkerk_rhs!(du, u, p, forcing, t)

Right-hand side of the *continuous* Rietkerk system, in method-of-lines form:

    ∂ₜO = d₁ ∇²O − l₁ O + R(t) − r₂ O B
    ∂ₜW = d₂ ∇²W − l₂ W + r₂ O B − r₁ W B
    ∂ₜB = d₃ ∇²B − l₃ B + j r₁ W B

`u` and `du` are `H×W×3` arrays whose third axis is `(O, W, B)`.

This is the model the hand-rolled [`step!`](@ref) *approximates* — but not the one
it implements, because that routine advances the fields sequentially and so feeds
each equation the already-updated value of the previous one. Here all three
derivatives are evaluated at the same state, as an ODE requires. The two agree in
the `dt → 0` limit; see `scripts/diffeq_comparison.jl` for the measured gap.

No scratch buffer is needed: each Laplacian is written into its own output slice
and then transformed in place, which also keeps the function allocation-free and
therefore safe under ForwardDiff and threading.
"""
function rietkerk_rhs!(du, u, p::RietkerkParams, forcing, t)
    O = @view u[:, :, 1]
    W = @view u[:, :, 2]
    B = @view u[:, :, 3]
    dO = @view du[:, :, 1]
    dW = @view du[:, :, 2]
    dB = @view du[:, :, 3]

    d1 = p.surface_water_diffusion_coeff
    d2 = p.soil_water_diffusion_coeff
    d3 = p.biomass_diffusion_coeff
    l1 = p.evaporation_rate
    l2 = p.seepage_rate
    l3 = p.mortality_rate
    r2 = p.infiltration_rate
    r1 = p.plant_uptake_rate
    j = p.water_use_efficiency
    R = forcing(t)

    laplacian!(dO, O)
    @. dO = d1 * dO - l1 * O + R - r2 * O * B

    laplacian!(dW, W)
    @. dW = d2 * dW - l2 * W + r2 * O * B - r1 * W * B

    laplacian!(dB, B)
    @. dB = d3 * dB - l3 * B + j * r1 * W * B

    return nothing
end

"""
    pack_state(surface_water, soil_water, biomass) -> Array{T,3}
    pack_state(s::SimState) -> Array{T,3}

Stack the three fields into the `H×W×3` layout the ODE backend uses.
"""
function pack_state(O::AbstractMatrix, W::AbstractMatrix, B::AbstractMatrix)
    size(O) == size(W) == size(B) ||
        throw(DimensionMismatch("all three fields must have the same size"))
    u = similar(O, size(O)..., 3)
    u[:, :, 1] .= O
    u[:, :, 2] .= W
    u[:, :, 3] .= B
    return u
end

pack_state(s::SimState) = pack_state(s.surface_water, s.soil_water, s.biomass)

"""
    pack_state(biomass; T = eltype(biomass)) -> Array{T,3}

Initial condition from a biomass field alone, with both water fields at zero —
the initialisation used throughout the study.
"""
function pack_state(biomass::AbstractMatrix; T::Type = eltype(biomass))
    b = convert.(T, biomass)
    return pack_state(zero(b), zero(b), b)
end

"""
    unpack_state(u) -> (surface_water, soil_water, biomass)

Views onto the three fields of a packed state. No copy.
"""
unpack_state(u::AbstractArray{<:Any,3}) =
    (view(u, :, :, 1), view(u, :, :, 2), view(u, :, :, 3))

"""
    biomass_of(u)

The biomass slice of a packed state — the field every diagnostic is computed from.
"""
biomass_of(u::AbstractArray{<:Any,3}) = view(u, :, :, 3)

# ---------------------------------------------------------------------------
# Methods supplied by the OrdinaryDiffEq extension.
# ---------------------------------------------------------------------------

"""
    solve_year(p, u0, weekly_precip, cfg; alg, kwargs...) -> ODESolution

Integrate one year of the continuous model with an adaptive solver.

Requires `using OrdinaryDiffEq` — the implementation lives in a package extension
so that the core package does not pay OrdinaryDiffEq's load time.

`ROCK2` is the recommended algorithm: the diffusion term is stiff (see
[`diffusion_stability_limit`](@ref)), and stabilised explicit Runge–Kutta methods
handle real-axis stiffness without forming a Jacobian, which matters at this state
size (three fields on a 131×140 grid is ~55 000 unknowns).
"""
function solve_year end

"""
    simulate_years_ode(p, biomass0, weekly_precip, cfg, nyears; alg, kwargs...)

Repeat [`solve_year`](@ref), returning `(final_state, mean_biomass_per_year)`.
Requires `using OrdinaryDiffEq`.
"""
function simulate_years_ode end

"""
    ode_loss(θ, prob; alg, kwargs...)

The delta-MSE objective of [`loss`](@ref), evaluated with the adaptive ODE backend
instead of the fixed-step scheme. Requires `using OrdinaryDiffEq`.
"""
function ode_loss end
