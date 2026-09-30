"""
    turing_value(p) -> Float64

Sign-valued diagnostic for the Turing (diffusion-driven) instability of the
homogeneous vegetated state:

    T = −d₁ l₂ l₃ + d₃ (l₁ + √(l₁ l₂ r₂ / r₁)) (l₂ + √(l₁ l₂ r₁ / r₂))

A **negative** value means the uniform state is unstable to spatial perturbations,
i.e. the parameter set can produce vegetation patterns; positive means it cannot.
Port of `compute_turing_evolution`.

Returns `NaN` if the parameters are non-positive in a way that makes the square
roots undefined.
"""
function turing_value(p::RietkerkParams)
    d1 = p.surface_water_diffusion_coeff
    d3 = p.biomass_diffusion_coeff
    l1 = p.evaporation_rate
    l2 = p.seepage_rate
    l3 = p.mortality_rate
    r1 = p.plant_uptake_rate
    r2 = p.infiltration_rate
    (l1 * l2 * r2 / r1 < 0 || l1 * l2 * r1 / r2 < 0) && return NaN
    return -d1 * l2 * l3 + d3 * (l1 + sqrt(l1 * l2 * r2 / r1)) * (l2 + sqrt(l1 * l2 * r1 / r2))
end

"""
    composite_value(p, B) -> Float64

Dimensionless growth efficiency at biomass level `B`:

    (r₂B / (l₁ + r₂B)) · (r₁B / (l₂ + r₁B)) · j

the product of the fraction of surface water that infiltrates, the fraction of
soil water that plants take up, and the water-use efficiency. Individually the
rate constants are only weakly identifiable from the data; this combination is
what the fits actually pin down, which is why it is reported alongside them.

Port of `compute_composite`. `B` is normally the mean training biomass — see
[`mean_training_biomass`](@ref).
"""
function composite_value(p::RietkerkParams, B::Real)
    l1 = p.evaporation_rate
    l2 = p.seepage_rate
    r1 = p.plant_uptake_rate
    r2 = p.infiltration_rate
    j = p.water_use_efficiency
    return (r2 * B / (l1 + r2 * B)) * (r1 * B / (l2 + r1 * B)) * j
end

"""
    mean_training_biomass(data_dir, sites; multiplier = 1500.0) -> Float64

Mean over all NDVI images of each image's mean biomass, for the given subsites.
This is the `B` that [`composite_value`](@ref) is evaluated at.
"""
function mean_training_biomass(data_dir::AbstractString, sites; multiplier::Real = 1500.0)
    means = Float64[]
    for site in sites
        name = startswith(String(site), "subsite_") ? String(site) : "subsite_$site"
        dir = joinpath(data_dir, name, "$(name)_ndvi")
        isdir(dir) || (@warn "NDVI directory not found; skipping" dir; continue)
        for f in sort(filter(x -> endswith(lowercase(x), ".tif"), readdir(dir)))
            push!(means, Statistics.mean(ndvi_biomass(joinpath(dir, f);
                                                      multiplier = multiplier, T = Float32)))
        end
    end
    isempty(means) && throw(ArgumentError("no NDVI images found under $data_dir for $sites"))
    return Statistics.mean(means)
end

"""
    tier1_filter(histories; length_frac = 0.9, degen_min = 1e-4) -> (kept, dropped)

Structural quality filter over a `model_id => parameter_history` collection,
matching `filter_tier1` in `bifurcation_parallel.py`.

A run is dropped when either
- its history is shorter than `length_frac` of the longest history — the run hit a
  non-finite loss and stopped early, or
- any final parameter is non-finite or has collapsed onto the clamp floor
  (`<= degen_min`) — the optimiser drove a coefficient to zero rather than fitting it.

`kept` is a vector of model ids; `dropped` is a vector of `(id, reason)` pairs, so
the exclusions can be reported rather than silently applied.
"""
function tier1_filter(histories::AbstractDict; length_frac::Real = 0.9, degen_min::Real = 1e-4)
    isempty(histories) && return (Int[], Tuple{Int,String}[])
    max_len = maximum(length, values(histories))
    min_len = length_frac * max_len

    kept = Int[]
    dropped = Tuple{Int,String}[]
    for id in sort(collect(keys(histories)))
        history = histories[id]
        reasons = String[]
        if length(history) < min_len
            push!(reasons, "short history ($(length(history)) < $(round(Int, min_len)) snapshots)")
        end
        final = history[end]
        bad = String[]
        for name in PARAM_NAMES
            v = get(final, name, NaN)
            if !isfinite(v)
                push!(bad, "$name=nan/inf")
            elseif v <= degen_min
                push!(bad, "$name=" * @sprintf("%.2e", v))
            end
        end
        isempty(bad) || push!(reasons, "degenerate final value(s): " * join(bad, ", "))
        isempty(reasons) ? push!(kept, id) : push!(dropped, (id, join(reasons, "; ")))
    end
    return (kept, dropped)
end

"""
    drop_degenerate(histories; threshold = 0.0011) -> (kept, dropped)

Weaker sibling of [`tier1_filter`](@ref): drop a run only if one of its final
parameters is below `threshold`, ignoring how long the run lasted.

This is the rule `realdata_parameter_analysis.py` applies, and it is deliberately
kept separate — the two Python scripts filter differently, so reproducing either
means choosing the matching rule rather than silently applying the stricter one.
A run that stopped after two snapshots but happens to have non-degenerate values
survives this filter and would not survive [`tier1_filter`](@ref).
"""
function drop_degenerate(histories::AbstractDict; threshold::Real = 0.0011)
    kept = Int[]
    dropped = Tuple{Int,String}[]
    for id in sort(collect(keys(histories)))
        history = histories[id]
        if isempty(history)
            push!(dropped, (id, "empty history"))
            continue
        end
        final = history[end]
        if !all(n -> haskey(final, n), PARAM_NAMES)
            push!(dropped, (id, "missing parameters in final snapshot"))
            continue
        end
        bad = [n for n in PARAM_NAMES if !isfinite(final[n]) || final[n] < threshold]
        isempty(bad) ? push!(kept, id) :
            push!(dropped, (id, "degenerate final value(s): " * join(bad, ", ")))
    end
    return (kept, dropped)
end

"""
    agreement_table(values; ground_truth = nothing) -> DataFrame

Cross-run agreement statistics for one quantity sampled over several fits.

`values` maps a quantity name to the vector of its fitted values, one per run.
For each quantity the table reports mean, standard deviation, coefficient of
variation, range, and — where a `ground_truth` value is supplied — the mean
absolute percentage error and the signed bias against it.

`agreement_score = 1 / (1 + cv)` is the same summary used in the Python analysis:
1 means every run agreed, 0 means they did not.

Port of `calculate_parameter_agreement`.
"""
function agreement_table(values::AbstractDict{<:AbstractString,<:AbstractVector};
                         ground_truth::Union{Nothing,RietkerkParams} = nothing)
    gt = ground_truth === nothing ? Dict{String,Float64}() : paramdict(ground_truth)
    rows = NamedTuple[]
    for name in sort(collect(keys(values)))
        v = Float64.(values[name])
        isempty(v) && continue
        mean_val = Statistics.mean(v)
        std_val = Statistics.std(v; corrected = false)
        cv = mean_val == 0 ? Inf : std_val / abs(mean_val)
        range_val = maximum(v) - minimum(v)
        gt_val = get(gt, name, NaN)
        mape = isnan(gt_val) ? NaN : Statistics.mean(abs.((v .- gt_val) ./ gt_val)) * 100
        bias = isnan(gt_val) ? NaN : Statistics.mean((v .- gt_val) ./ gt_val) * 100
        push!(rows, (
            parameter = name,
            n_models = length(v),
            mean = mean_val,
            std = std_val,
            cv = cv,
            min = minimum(v),
            max = maximum(v),
            range = range_val,
            rel_range_pct = mean_val == 0 ? Inf : range_val / abs(mean_val) * 100,
            ground_truth = gt_val,
            mape_vs_gt_pct = mape,
            bias_vs_gt_pct = bias,
            agreement_score = isfinite(cv) ? 1 / (1 + cv) : 0.0,
        ))
    end
    return DataFrames.DataFrame(rows)
end

"""
    bifurcation_sweep(params, initial_biomass, precip_values; years, cfg,
                      snapshot_at = Float64[], forcing = summer_weekly_precip)
        -> (means, snapshots)

Equilibrium mean biomass as a function of annual rainfall, for one parameter set.

Each rainfall level is simulated independently for `years` years from the same
initial field; the reported value is the mean biomass of the final year.

!!! note "Single branch, so no hysteresis"
    Every level starts from the same `initial_biomass`. Sweeping upward from a bare
    state and downward from a vegetated one would reveal whether the transition is
    bistable — the question a bifurcation diagram usually exists to answer. This
    function reproduces the single-branch sweep of `bifurcation_parallel.py`; to
    look for hysteresis, call it twice with different initial fields.

`forcing(annual_mm)` builds the weekly profile; the default is the summer-pulse
form the published diagram uses.

Levels are independent, so the sweep runs under `Threads.@threads`; start Julia
with `-t auto` to use it.

`snapshot_at` lists rainfall values whose final biomass *field* should also be
returned, for the spatial insets on the diagram.
"""
function bifurcation_sweep(p::RietkerkParams, initial_biomass::AbstractMatrix,
                           precip_values::AbstractVector; years::Integer, cfg::SimConfig,
                           snapshot_at::AbstractVector = Float64[],
                           forcing = summer_weekly_precip)
    means = zeros(Float64, length(precip_values))
    snapshots = Dict{Float64,Matrix{Float64}}()
    lk = ReentrantLock()
    Threads.@threads for i in eachindex(precip_values)
        precip = precip_values[i]
        state = simstate(initial_biomass)
        simulate_years!(state, p, forcing(precip), cfg, years)
        means[i] = Statistics.mean(state.biomass)
        if any(v -> isapprox(v, precip; atol = 1e-9), snapshot_at)
            field = Float64.(state.biomass)
            lock(lk) do
                snapshots[Float64(precip)] = field
            end
        end
    end
    return (means, snapshots)
end
