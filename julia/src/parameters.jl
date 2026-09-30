"""
    RietkerkParams{T}

The nine scalar ecological coefficients that are fitted by the inverse problem.

Field order is fixed and matches [`PARAM_NAMES`](@ref), which in turn matches the
column order of `four_site_final_parameter_values.csv` and the `named_parameters()`
keys written by the original PyTorch code — so parameter vectors round-trip between
the two implementations without reordering.

The struct is parameterised on `T` so that it can hold `Float64`, `Float32`, or
`ForwardDiff.Dual` values; the last is what makes the model differentiable.
"""
struct RietkerkParams{T<:Real}
    surface_water_diffusion_coeff::T
    soil_water_diffusion_coeff::T
    biomass_diffusion_coeff::T
    evaporation_rate::T
    seepage_rate::T
    mortality_rate::T
    infiltration_rate::T
    plant_uptake_rate::T
    water_use_efficiency::T
end

"""
Names of the fitted parameters, in struct/vector order.

These strings are the wire format shared with the Python code (CSV headers, JSON
keys, `state_dict` entries), so do not rename them.
"""
const PARAM_NAMES = (
    "surface_water_diffusion_coeff",
    "soil_water_diffusion_coeff",
    "biomass_diffusion_coeff",
    "evaporation_rate",
    "seepage_rate",
    "mortality_rate",
    "infiltration_rate",
    "plant_uptake_rate",
    "water_use_efficiency",
)

const NPARAMS = length(PARAM_NAMES)

RietkerkParams(vals::NTuple{9,T}) where {T<:Real} = RietkerkParams{T}(vals...)

function RietkerkParams(v::AbstractVector{T}) where {T<:Real}
    length(v) == NPARAMS ||
        throw(ArgumentError("expected $NPARAMS parameters, got $(length(v))"))
    RietkerkParams{T}(ntuple(i -> @inbounds(v[i]), NPARAMS)...)
end

"""
    RietkerkParams(; kwargs...)

Keyword constructor. Every parameter must be supplied; there is no silent default,
because a silently defaulted coefficient is indistinguishable from a fitted one.
"""
function RietkerkParams(;
    surface_water_diffusion_coeff,
    soil_water_diffusion_coeff,
    biomass_diffusion_coeff,
    evaporation_rate,
    seepage_rate,
    mortality_rate,
    infiltration_rate,
    plant_uptake_rate,
    water_use_efficiency,
)
    vals = promote(
        surface_water_diffusion_coeff,
        soil_water_diffusion_coeff,
        biomass_diffusion_coeff,
        evaporation_rate,
        seepage_rate,
        mortality_rate,
        infiltration_rate,
        plant_uptake_rate,
        water_use_efficiency,
    )
    RietkerkParams(vals)
end

Base.eltype(::RietkerkParams{T}) where {T} = T
Base.eltype(::Type{RietkerkParams{T}}) where {T} = T

"""
    paramvector(p) -> Vector{T}

Flatten to the vector layout used by the optimiser and by [`ForwardDiff`](@ref).
"""
paramvector(p::RietkerkParams{T}) where {T} =
    T[getfield(p, i) for i in 1:NPARAMS]

"""
    paramdict(p) -> Dict{String,Float64}

Convert to the string-keyed dictionary used in JSON/CSV output, matching the
Python `{name: param.item()}` snapshots.
"""
paramdict(p::RietkerkParams) =
    Dict{String,Float64}(PARAM_NAMES[i] => Float64(getfield(p, i)) for i in 1:NPARAMS)

"""
    RietkerkParams(d::AbstractDict)

Build from a string-keyed dictionary (e.g. one row of a parameter CSV, or a
snapshot loaded from the Python parameter history).
"""
function RietkerkParams(d::AbstractDict)
    vals = ntuple(NPARAMS) do i
        name = PARAM_NAMES[i]
        haskey(d, name) || throw(KeyError(name))
        Float64(d[name])
    end
    RietkerkParams(vals)
end

Base.convert(::Type{RietkerkParams{T}}, p::RietkerkParams) where {T} =
    RietkerkParams(ntuple(i -> T(getfield(p, i)), NPARAMS))

"""
    randparams([rng]; lo = 0.0, hi = 1.0) -> RietkerkParams{Float64}

Random initialisation for a training run. The original code used `torch.rand(1)`
per parameter, i.e. `Uniform[0, 1)`, which is the default here.

Note that exact reproduction of a PyTorch seed is not possible — the RNG streams
differ — so runs are reproducible within Julia, not across the two languages.
"""
function randparams(rng::Random.AbstractRNG = Random.default_rng(); lo = 0.0, hi = 1.0)
    RietkerkParams(ntuple(_ -> lo + (hi - lo) * rand(rng), NPARAMS))
end

"""
Ground-truth coefficients used to generate the synthetic experiments
(`train_invPDE_synthetic_batch.py`).
"""
const SYNTHETIC_TRUTH = RietkerkParams(
    surface_water_diffusion_coeff = 8.0,
    soil_water_diffusion_coeff = 1.0,
    biomass_diffusion_coeff = 0.05,
    evaporation_rate = 0.3,
    seepage_rate = 0.4,
    mortality_rate = 0.3,
    infiltration_rate = 0.1,
    plant_uptake_rate = 0.15,
    water_use_efficiency = 0.35,
)

"""
Ground-truth coefficients for the *single-site* synthetic experiment
(`train_invPDE_synthetic_batch_1site.py`).

The loss, evaporation, seepage, infiltration and uptake rates are all larger than
in [`SYNTHETIC_TRUTH`](@ref), and that script also integrates with
`year_time_units = 1.0` rather than `1.5` — so the two synthetic experiments are
not comparable parameter for parameter.
"""
const SYNTHETIC_TRUTH_1SITE = RietkerkParams(
    surface_water_diffusion_coeff = 8.0,
    soil_water_diffusion_coeff = 1.0,
    biomass_diffusion_coeff = 0.05,
    evaporation_rate = 0.6,
    seepage_rate = 0.8,
    mortality_rate = 0.6,
    infiltration_rate = 0.2,
    plant_uptake_rate = 0.35,
    water_use_efficiency = 0.35,
)

"""
Reference coefficients quoted as "ground truth" in the real-data analysis
(`realdata_train_invPDE.py` / `realdata_parameter_analysis.py`). These differ from
[`SYNTHETIC_TRUTH`](@ref) in the mortality, infiltration, uptake and efficiency
terms.
"""
const REALDATA_REFERENCE = RietkerkParams(
    surface_water_diffusion_coeff = 8.0,
    soil_water_diffusion_coeff = 1.0,
    biomass_diffusion_coeff = 0.05,
    evaporation_rate = 0.3,
    seepage_rate = 0.4,
    mortality_rate = 0.6,
    infiltration_rate = 2.1,
    plant_uptake_rate = 1.9,
    water_use_efficiency = 0.55,
)

function Base.show(io::IO, ::MIME"text/plain", p::RietkerkParams{T}) where {T}
    println(io, "RietkerkParams{", T, "}:")
    w = maximum(length, PARAM_NAMES)
    for i in 1:NPARAMS
        @printf(io, "  %-*s  %.6g\n", w, PARAM_NAMES[i], Float64(getfield(p, i)))
    end
end
