#!/usr/bin/env julia
"""
Time the Julia fit against the recorded PyTorch/A100 numbers.

Counterpart of `measure_runtime.py`, which produced `runtime_estimates.csv` on an
A100. This script times a handful of epochs the same way and extrapolates to a
full run, so the two are directly comparable.

Usage
-----
    julia --project=julia -t auto julia/scripts/benchmark.jl

Options
    --epochs 3        timed epochs per configuration (after one warm-up)
    --chunk-scan      also time every ForwardDiff chunk width
    --out <csv>

Reading the result: the comparison is CPU-Julia against GPU-PyTorch, so it says
nothing about language speed in isolation. What it does show is that this problem
— a 130x140 grid, nine parameters, thousands of sequential tiny kernels — is a bad
fit for a GPU, and that forward-mode AD over nine parameters beats taping the
rollout.
"""

using InverseTuring
using Printf
using Random
import CSV, DataFrames, Statistics

include(joinpath(@__DIR__, "common.jl"))

const TIMED = argint("epochs", 3)
const CHUNK_SCAN = argflag("chunk-scan")
const OUTCSV = string(argval("out", joinpath(OUT_ROOT, "runtime_estimates.csv")))

# Recorded PyTorch timings, from runtime_estimates.csv (NVIDIA A100-SXM4-40GB).
const PYTORCH = Dict(
    "invPDE_real_4site" => (epochs = 7500, sec_per_epoch = 12.1013),
    "invPDE_synthetic_4site" => (epochs = 7500, sec_per_epoch = 3.7951),
    "invPDE_synthetic_1site" => (epochs = 7500, sec_per_epoch = 0.9661),
)

hms(s) = (h = floor(Int, s / 3600); m = floor(Int, (s - 3600h) / 60);
          @sprintf("%dh %02dm %02ds", h, m, round(Int, s - 3600h - 60m)))

banner("Benchmark: Julia (CPU) vs recorded PyTorch (A100)")
nthreads = report_threads()
println("timed epochs per configuration: ", TIMED)

"""
Time `TIMED` gradient evaluations after a warm-up, returning the **fastest**
seconds/epoch.

Minimum rather than mean: on a shared desktop, background load only ever adds
time, so the minimum is the closest estimate of the cost of the work itself.
Means here drifted by more than 2x depending on what else was running.
"""
function time_epochs(prob, θ; chunk = DEFAULT_CHUNK)
    loss_and_gradient(prob, θ; chunk_size = chunk)     # warm up / compile
    best = Inf
    for _ in 1:TIMED
        t0 = time()
        loss_and_gradient(prob, θ; chunk_size = chunk)
        best = min(best, time() - t0)
    end
    return best
end

rows = NamedTuple[]

# ---------------------------------------------------------------------------
banner("invPDE real data, 4 sites")
sites = load_sites(DATA_DIR, ["b", "i", "c", "e"]; multiplier = 1500.0, T = Float64)
cfg = SimConfig(steps_per_week = 3, year_time_units = 1.0)
θ = paramvector(randparams(Xoshiro(1)))
for threaded in (false, true)
    prob = InverseProblem(sites, cfg; threaded = threaded)
    spe = time_epochs(prob, θ)
    push!(rows, (experiment = "invPDE_real_4site", threading = threaded ? "sites" : "serial",
                 threads = threaded ? nthreads : 1, target_epochs = 7500,
                 mean_epoch_seconds = spe, estimated_full_run_seconds = spe * 7500))
    @printf("  %-8s %7.3f s/epoch  ->  %s for 7500 epochs\n",
            threaded ? "threaded" : "serial", spe, hms(spe * 7500))
end

# ---------------------------------------------------------------------------
banner("invPDE synthetic, 4 sites and 1 site")
for (name, nsites, truth, ytu, rain, arange, eprecip) in (
        ("invPDE_synthetic_4site", 4, SYNTHETIC_TRUTH, 1.5, 0.75, (15.0, 27.0), 21.0),
        ("invPDE_synthetic_1site", 1, SYNTHETIC_TRUTH_1SITE, 1.0, 1.0, (16.5, 24.5), 19.0))
    scfg = SimConfig(steps_per_week = 1, year_time_units = ytu)
    # A 20-year spin-up instead of 100 keeps setup short; it does not affect the
    # per-epoch cost, which is what is being measured.
    data = synthetic_experiment(; p = truth, grid = (128, 128), n_sites = nsites,
                                rain_multiplier = rain, annual_range = arange,
                                equilibrium_precip = eprecip, equilibrium_years = 20,
                                years = 10, cfg = scfg, rng = Xoshiro(42))
    for threaded in (nsites > 1 ? (false, true) : (false,))
        prob = InverseProblem(data.problem.trajectories, scfg;
                              average = false, threaded = threaded)
        spe = time_epochs(prob, θ)
        push!(rows, (experiment = name, threading = threaded ? "sites" : "serial",
                     threads = threaded ? nthreads : 1, target_epochs = 7500,
                     mean_epoch_seconds = spe, estimated_full_run_seconds = spe * 7500))
        @printf("  %-24s %-8s %7.3f s/epoch  ->  %s\n", name,
                threaded ? "threaded" : "serial", spe, hms(spe * 7500))
    end
end

# ---------------------------------------------------------------------------
if CHUNK_SCAN
    banner("ForwardDiff chunk width scan (real data, serial)")
    prob = InverseProblem(sites, cfg; threaded = false)
    for c in (1, 2, 3, 5, 9)
        spe = time_epochs(prob, θ; chunk = c)
        @printf("  chunk %d: %7.3f s/epoch%s\n", c, spe, c == DEFAULT_CHUNK ? "   <- default" : "")
    end
end

# ---------------------------------------------------------------------------
banner("Comparison with the recorded A100 run")
@printf("%-26s %14s %14s %10s\n", "experiment", "PyTorch A100", "Julia CPU", "speedup")
println("-" ^ 68)
for (name, ref) in sort(collect(PYTORCH); by = first)
    mine = filter(r -> r.experiment == name, rows)
    isempty(mine) && continue
    best = minimum(r -> r.mean_epoch_seconds, mine)
    @printf("%-26s %11.3f s %11.3f s %9.2fx\n", name, ref.sec_per_epoch, best,
            ref.sec_per_epoch / best)
end
println("-" ^ 68)
println("(s/epoch, best of $TIMED; speedup > 1 means Julia on this CPU beats PyTorch on the A100)")
println("Note: the 1-site synthetic case is the one a GPU handles well — a single")
println("large batch of work per kernel. The multi-site cases are where launch")
println("overhead dominates and the CPU wins.")

df = DataFrames.DataFrame(rows)
df.estimated_full_run_hms = hms.(df.estimated_full_run_seconds)
df.device = fill("cpu-$(Sys.CPU_NAME)", DataFrames.nrow(df))
mkpath(dirname(OUTCSV))
CSV.write(OUTCSV, df)
println("\nWritten to ", OUTCSV)
