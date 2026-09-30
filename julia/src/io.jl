"""
    load_parameter_history(path) -> Vector{Dict{String,Float64}}

Read a parameter history written either by this package (`.json`) or by the
original PyTorch code (`.pkl`, a pickled `List[Dict[str, float]]`).

Reading the Python pickles directly means existing runs under
`results/real_data/models/parameters/` can be analysed with the Julia tooling
without re-fitting anything.
"""
function load_parameter_history(path::AbstractString)
    isfile(path) || throw(ArgumentError("no such file: $path"))
    raw = if endswith(lowercase(path), ".pkl")
        open(Pickle.load, path)
    else
        JSON.parsefile(path)
    end
    return [Dict{String,Float64}(String(k) => Float64(v) for (k, v) in snap) for snap in raw]
end

"""
    load_parameter_histories(dir; pattern = r"model_(\\d+)_params\\.(pkl|json)") -> Dict{Int,Vector{Dict}}

Load every parameter history in a directory, keyed by the model id embedded in the
filename. Feeds [`tier1_filter`](@ref).
"""
function load_parameter_histories(dir::AbstractString;
                                  pattern::Regex = r"model_(\d+)_params\.(pkl|json)$")
    isdir(dir) || throw(ArgumentError("no such directory: $dir"))
    out = Dict{Int,Vector{Dict{String,Float64}}}()
    for f in readdir(dir)
        m = match(pattern, f)
        m === nothing && continue
        out[parse(Int, m.captures[1])] = load_parameter_history(joinpath(dir, f))
    end
    return out
end

"""
    read_parameter_table(path) -> DataFrame

Read a parameter CSV such as `four_site_final_parameter_values.csv`. The unnamed
index column is renamed `model_id`, so rows can be matched against test metrics
and parameter histories.
"""
function read_parameter_table(path::AbstractString)
    df = CSV.read(path, DataFrames.DataFrame)
    first_col = DataFrames.names(df)[1]
    if isempty(first_col) || first_col == "Column1"
        DataFrames.rename!(df, first_col => "model_id")
    end
    return df
end

"""
    params_from_row(row) -> RietkerkParams{Float64}

Extract the nine coefficients from one row of a parameter table.
"""
params_from_row(row) = RietkerkParams(Dict(n => Float64(row[n]) for n in PARAM_NAMES))

"""
    write_parameter_table(path, ids, params; extra = Dict())

Write fitted parameters to CSV in the same column order as the Python output, so
the existing Python analysis scripts can consume Julia results unchanged.
`extra` adds derived columns (e.g. `"turing_value" => values`).
"""
function write_parameter_table(path::AbstractString, ids::AbstractVector,
                               params::AbstractVector{<:RietkerkParams};
                               extra::AbstractDict = Dict{String,Vector{Float64}}())
    length(ids) == length(params) ||
        throw(DimensionMismatch("ids and params must have equal length"))
    df = DataFrames.DataFrame("model_id" => collect(ids))
    for (i, name) in enumerate(PARAM_NAMES)
        df[!, name] = [Float64(getfield(p, i)) for p in params]
    end
    for (name, values) in extra
        df[!, name] = collect(values)
    end
    mkpath(dirname(path))
    CSV.write(path, df)
    return path
end

"""
    save_run(path, result; metadata = Dict())

Serialise a [`TrainResult`](@ref) to JSON, using the same key names as the Python
`result_dict` (`initial_params`, `parameter_history`, `loss_history`, ...).
"""
function save_run(path::AbstractString, r::TrainResult; metadata::AbstractDict = Dict{String,Any}())
    payload = Dict{String,Any}(
        "seed" => r.seed,
        "initial_params" => paramdict(r.initial_params),
        "final_params" => paramdict(r.params),
        "parameter_history" => r.parameter_history,
        "loss_history" => r.loss_history,
        "num_epochs" => r.epochs_run,
        "final_loss" => r.final_loss,
        "elapsed_seconds" => r.elapsed_seconds,
        "converged" => r.converged,
    )
    merge!(payload, Dict{String,Any}(String(k) => v for (k, v) in metadata))
    return save_json(path, payload)
end

"""
    load_run(path) -> Dict{String,Any}

Read back a run written by [`save_run`](@ref), or a `result_*.json` produced by the
Python batch scripts — the schemas match.
"""
load_run(path::AbstractString) = JSON.parsefile(path; allownan = true)

"""
    save_json(path, obj)

Write any JSON-serialisable object, creating parent directories as needed.

`NaN` and `Inf` are written literally rather than rejected. Strict JSON has no
representation for them, but a diverged fit legitimately produces a NaN loss and
the Python side wrote those values too — refusing them here would mean losing the
record of exactly the runs worth inspecting. Readers that reject them (including
`JSON.parsefile` without `allownan`) will need the same flag.
"""
function save_json(path::AbstractString, obj)
    mkpath(dirname(path))
    write(path, JSON.json(obj; allownan = true, pretty = 2))
    return path
end
