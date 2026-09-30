"""
    ndvi_biomass(path; multiplier = 1500.0, T = Float64) -> Matrix{T}

Read a two-band (Red, NIR) GeoTIFF and convert it to the biomass proxy used for
fitting:

    NDVI    = (NIR − Red) / (NIR + Red)   where NIR + Red > 0, else 0
    biomass = clamp(max(NDVI, 0) * multiplier, 0, multiplier)

Band 1 is Red and band 2 is NIR, matching the export convention of the files in
`data/`. Because NDVI is a ratio, any common linear scaling of the two bands
cancels — the raw digital numbers can stay unscaled.

The result is transposed relative to GDAL's native `(x, y)` layout so that rows
are image rows, matching `rasterio`/`numpy` and therefore the orientation of every
figure produced by the Python code.

`T = Float32` reproduces the original PyTorch arithmetic bit-for-bit; the default
`Float64` is a better base for gradient-based fitting.
"""
function ndvi_biomass(path::AbstractString; multiplier::Real = 1500.0, T::Type = Float64)
    isfile(path) || throw(ArgumentError("no such raster: $path"))
    mult = T(multiplier)
    return ArchGDAL.read(path) do ds
        nb = ArchGDAL.nraster(ds)
        nb == 2 || throw(ArgumentError(
            "expected 2 bands (Red, NIR), found $nb in $path"))
        # GDAL hands back (x, y); permute to (row, col).
        red = permutedims(T.(ArchGDAL.read(ds, 1)))
        nir = permutedims(T.(ArchGDAL.read(ds, 2)))
        biomass = similar(red)
        @inbounds for i in eachindex(red, nir, biomass)
            den = nir[i] + red[i]
            ndvi = den > 0 ? (nir[i] - red[i]) / den : zero(T)
            biomass[i] = clamp(max(ndvi, zero(T)) * mult, zero(T), mult)
        end
        biomass
    end
end

"""
    YearObservation{T}

One satellite observation: the biomass field for a year, together with the
precipitation that drives the model from this year to the next.
"""
struct YearObservation{T}
    year::Int
    biomass::Matrix{T}
    precipitation::Float64
    weekly_precipitation::Vector{Float64}
end

"""
    SiteSeries{T}

A site's observations, sorted by year. `name` is the directory name
(e.g. `"subsite_b"`), matching the keys used in the Python results.
"""
struct SiteSeries{T}
    name::String
    observations::Vector{YearObservation{T}}
end

Base.length(s::SiteSeries) = length(s.observations)
Base.getindex(s::SiteSeries, i) = s.observations[i]
Base.iterate(s::SiteSeries, st...) = iterate(s.observations, st...)
years(s::SiteSeries) = [o.year for o in s.observations]
Base.size(s::SiteSeries) = isempty(s.observations) ? (0, 0) : size(s.observations[1].biomass)

function Base.show(io::IO, s::SiteSeries{T}) where {T}
    ys = years(s)
    dims = isempty(s.observations) ? "empty" : join(size(s.observations[1].biomass), "x")
    rng = isempty(ys) ? "-" : "$(minimum(ys))-$(maximum(ys))"
    print(io, "SiteSeries{$T}(\"", s.name, "\", ", length(ys), " years ", rng, ", ", dims, ")")
end

"""
    read_annual_precip(path) -> Dict{Int,Float64}

Parse `<site>_precip.csv` (ERA5 annual totals).

The `datetime` column is reduced to its last four characters to recover the year
— the export sometimes prefixes an index (`232013` → `2013`). Totals are stored
in metres and converted to millimetres here.
"""
function read_annual_precip(path::AbstractString)
    df = CSV.read(path, DataFrames.DataFrame)
    cols = DataFrames.names(df)
    if "datetime" in cols && "total_precipitation_sum" in cols
        yrs = [parse(Int, last(string(v), 4)) for v in df.datetime]
        return Dict{Int,Float64}(zip(yrs, Float64.(df.total_precipitation_sum) .* 1000))
    end
    # Fall back to the Python loader's column sniffing.
    datecol = findfirst(c -> occursin("date", lowercase(c)) || occursin("year", lowercase(c)), cols)
    precipcol = findfirst(c -> occursin("precip", lowercase(c)) || occursin("rain", lowercase(c)), cols)
    (datecol === nothing || precipcol === nothing) &&
        throw(ArgumentError("could not identify date and precipitation columns in $path"))
    dvals = df[!, cols[datecol]]
    yrs = eltype(dvals) <: Number && maximum(dvals) <= 3000 ?
          Int.(dvals) : [parse(Int, last(string(v), 4)) for v in dvals]
    return Dict{Int,Float64}(zip(yrs, Float64.(df[!, cols[precipcol]])))
end

"""
    read_weekly_precip(path) -> Dict{Int,Vector{Float64}}

Parse `<site>_weekly_precip.csv` into per-year vectors of mm/day rates, ordered
by week.

Values are passed through untouched, including the small negative rates that
ERA5's accumulation differencing produces in dry weeks — the Python code feeds
those to the model as-is, and clipping them here would change the fit.
"""
function read_weekly_precip(path::AbstractString)
    df = CSV.read(path, DataFrames.DataFrame)
    out = Dict{Int,Vector{Float64}}()
    for sub in DataFrames.groupby(df, :year)
        ordered = sort(sub, :week)
        out[Int(first(ordered.year))] = Float64.(ordered.precipitation_mm_per_day)
    end
    return out
end

"""
    load_site(data_dir, site; multiplier = 1500.0, T = Float64,
              use_weekly_precip = true) -> SiteSeries{T}

Load one subsite. `site` may be the bare letter (`"b"`) or the full directory
name (`"subsite_b"`).

Years present as a raster but missing from the precipitation table are skipped
with a warning, as in the Python loader. When `use_weekly_precip` is true but a
year has no weekly record, the annual total is spread flat across 52 weeks.
"""
function load_site(data_dir::AbstractString, site::AbstractString;
                   multiplier::Real = 1500.0, T::Type = Float64,
                   use_weekly_precip::Bool = true)
    name = startswith(site, "subsite_") ? String(site) : "subsite_$site"
    dir = joinpath(data_dir, name)
    isdir(dir) || throw(ArgumentError("site directory not found: $dir"))

    precip_dir = joinpath(dir, "$(name)_precip")
    annual_file = joinpath(precip_dir, "$(name)_precip.csv")
    isfile(annual_file) || throw(ArgumentError("precipitation file not found: $annual_file"))
    annual = read_annual_precip(annual_file)

    weekly = Dict{Int,Vector{Float64}}()
    if use_weekly_precip
        weekly_file = joinpath(precip_dir, "$(name)_weekly_precip.csv")
        if isfile(weekly_file)
            weekly = read_weekly_precip(weekly_file)
        else
            @warn "weekly precipitation file not found; falling back to flat annual totals" file=weekly_file
        end
    end

    ndvi_dir = joinpath(dir, "$(name)_ndvi")
    isdir(ndvi_dir) || throw(ArgumentError("NDVI directory not found: $ndvi_dir"))
    rasters = filter(f -> endswith(lowercase(f), ".tif") || endswith(lowercase(f), ".tiff"),
                     readdir(ndvi_dir))

    obs = YearObservation{T}[]
    for f in rasters
        stem = first(splitext(f))
        year = tryparse(Int, last(split(stem, '_')))
        if year === nothing
            @warn "could not extract year from filename; skipping" file=f
            continue
        end
        if !haskey(annual, year)
            @warn "no precipitation data for year; skipping" site=name year=year
            continue
        end
        biomass = ndvi_biomass(joinpath(ndvi_dir, f); multiplier = multiplier, T = T)
        wk = use_weekly_precip ?
             get(weekly, year, uniform_weekly_precip(annual[year])) :
             uniform_weekly_precip(annual[year])
        push!(obs, YearObservation{T}(year, biomass, annual[year], wk))
    end

    sort!(obs, by = o -> o.year)
    return SiteSeries{T}(name, obs)
end

"""
    load_sites(data_dir, sites; kwargs...) -> Vector{SiteSeries}

Load several subsites, preserving the order given in `sites`. Pass `nothing` to
load every `subsite_*` directory found under `data_dir`.

Keyword arguments are forwarded to [`load_site`](@ref).
"""
function load_sites(data_dir::AbstractString, sites; kwargs...)
    names = if sites === nothing
        sort(filter(d -> startswith(d, "subsite_") && isdir(joinpath(data_dir, d)),
                    readdir(data_dir)))
    else
        collect(sites)
    end
    return [load_site(data_dir, s; kwargs...) for s in names]
end

"""
    biomass_stats(sites) -> NamedTuple

Per-image mean biomass summarised across sites, matching the `global_stats`
block that `realdata_train_invPDE.py` writes to `data_info.json`. Useful as a
data-integrity check against an existing Python run.
"""
function biomass_stats(sites::AbstractVector{<:SiteSeries})
    means = Float64[]
    precips = Float64[]
    for s in sites, o in s.observations
        push!(means, Statistics.mean(o.biomass))
        push!(precips, o.precipitation)
    end
    isempty(means) && throw(ArgumentError("no observations loaded"))
    return (
        min_biomass = minimum(means), max_biomass = maximum(means),
        mean_biomass = Statistics.mean(means), std_biomass = Statistics.std(means; corrected = false),
        min_precip = minimum(precips), max_precip = maximum(precips),
        mean_precip = Statistics.mean(precips), std_precip = Statistics.std(precips; corrected = false),
        n_images = length(means),
    )
end
