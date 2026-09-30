"""
Shared helpers for the command-line scripts: minimal argument parsing and the
repository paths every script needs.

Kept dependency-free on purpose — the scripts are meant to be readable end to end,
and an argument-parsing package would be more machinery than nine flags deserve.
"""

const REPO = normpath(joinpath(@__DIR__, "..", ".."))

"""Satellite rasters and precipitation CSVs. Read-only."""
const DATA_DIR = joinpath(REPO, "data")

"""
Existing PyTorch results. **Read-only** — scripts load parameter tables, metrics
and histories from here but never write to it, so a Julia run can never overwrite
a published figure or table.
"""
const PY_RESULTS = joinpath(REPO, "results")

"""
Where every Julia script writes. Kept inside `julia/` so the port is entirely
self-contained and `results/` stays exactly as the Python code left it.
"""
const OUT_ROOT = joinpath(REPO, "julia", "results")

"""
    argval(name, default) -> String

Read `--name value` or `--name=value` from `ARGS`, falling back to `default`.
"""
function argval(name::AbstractString, default)
    flag = "--$name"
    for (i, a) in enumerate(ARGS)
        a == flag && i < length(ARGS) && return ARGS[i + 1]
        startswith(a, flag * "=") && return a[(length(flag) + 2):end]
    end
    return default
end

argint(name, default::Integer) = parse(Int, string(argval(name, default)))
argfloat(name, default::Real) = parse(Float64, string(argval(name, default)))
argflag(name) = "--$name" in ARGS

"""
    arglist(name, default::Vector{String}) -> Vector{String}

Comma-separated list argument, e.g. `--sites b,i,c,e`.
"""
function arglist(name::AbstractString, default::Vector{String})
    raw = argval(name, nothing)
    raw === nothing && return default
    return String.(filter(!isempty, strip.(split(string(raw), ','))))
end

"""
    banner(title)

Section header, matching the style of the Python console output so logs from the
two implementations read the same.
"""
function banner(title::AbstractString)
    println("\n" * "=" ^ 72)
    println(" ", title)
    println("=" ^ 72)
    flush(stdout)
end

"""
    report_threads()

Print the thread count, and warn when Julia was started single-threaded — the
fits and sweeps are the kind of job where forgetting `-t auto` costs hours.
"""
function report_threads()
    n = Threads.nthreads()
    println("Julia threads: $n  (CPU threads available: $(Sys.CPU_THREADS))")
    n == 1 && @warn "running single-threaded; restart with `julia -t auto` for a large speedup"
    return n
end
