#  Parameter trajectories along the L-BFGS path.
#
#  Reads the incremental `*_trajectory.csv` files written by the L-BFGS runner
#  (one row per objective evaluation), so it also works on a run in progress.
#
#  Each parameter gets its own panel with its own y-limits. Sharing a scale
#  across parameters is actively misleading here: the fitted coefficients span
#  four orders of magnitude, so a common axis collapses mortality (~0.08) and
#  biomass diffusion (~0.01) onto a flat line while evaporation (~200) uses the
#  whole panel.
#
#  Usage:
#     julia --project=julia julia/scripts/plot_lbfgs_trajectories.jl <traj_dir> [outdir] [--seeds a,b,c]
#
#  `--seeds` restricts the plot to named seeds. A 30-model run puts 30 lines in
#  every panel, which is unreadable; the flag is how you look at a few.

using InverseTuring, Plots, Printf
import CSV, DataFrames, Statistics

const SEEDFILTER = let i = findfirst(==("--seeds"), ARGS)
    i === nothing ? nothing : Set(parse.(Int, split(ARGS[i + 1], ",")))
end
# Axes are linear. `--log` switches wide-range panels to log10, which un-squashes
# the parameters that travel two decades from their random start but makes the
# vertical distance between two runs no longer mean what it looks like.
const ALLOW_LOG = "--log" in ARGS

const POS = let a = copy(ARGS), i = findfirst(==("--seeds"), a)
    i === nothing || deleteat!(a, [i, i + 1])
    filter(!=("--log"), a)
end

const TRAJDIR = length(POS) >= 1 ? POS[1] :
    error("usage: plot_lbfgs_trajectories.jl <dir with *_trajectory.csv> [outdir] [--seeds a,b]")
const OUTDIR = length(POS) >= 2 ? POS[2] :
    joinpath(@__DIR__, "..", "results", "lbfgs_trajectories")
mkpath(OUTDIR)

files = sort(filter(f -> endswith(f, "_trajectory.csv"), readdir(TRAJDIR; join = true)))
if SEEDFILTER !== nothing
    files = filter(f -> parse(Int, match(r"seed_0*(\d+)_", basename(f)).captures[1]) in SEEDFILTER,
                   files)
end
isempty(files) && error("no matching *_trajectory.csv in $TRAJDIR")

#  Two on-disk layouts are read. The standalone L-BFGS runner writes an `eval`
#  column and nothing else; `train_hybrid.jl` writes `phase` and `step`, where
#  `step` restarts at 1 when the Adam warm-up hands over to L-BFGS. Plotting
#  `step` directly would fold the two phases on top of each other, so the x-axis
#  is a running row index in both cases and the handover is drawn as a rule.
function read_trajectory(f)
    df = CSV.read(f, DataFrames.DataFrame)
    cols = names(df)
    df.x = collect(1:DataFrames.nrow(df))
    handover = if "phase" in cols
        i = findfirst(==("lbfgs"), df.phase)
        i === nothing ? nothing : df.x[i]
    else
        nothing
    end
    return df, handover
end

# A model that has only written its `init` row contributes a single point at a
# Uniform[0,1) draw, which for a coefficient fitted near 0.01 is two orders of
# magnitude off and would set the panel's limits on its own. Skip runs that have
# not actually started.
const MIN_ROWS = 5

runs = NamedTuple[]
for f in files
    seed = parse(Int, match(r"seed_0*(\d+)_", basename(f)).captures[1])
    df, handover = read_trajectory(f)
    if DataFrames.nrow(df) < MIN_ROWS
        @printf("seed %-5d skipped (%d rows, not started)\n", seed, DataFrames.nrow(df))
        continue
    end
    push!(runs, (seed = seed, df = df, handover = handover))
end
isempty(runs) && error("no run in $TRAJDIR has reached $MIN_ROWS rows yet")
for r in runs
    ok = filter(isfinite, r.df.loss)
    @printf("seed %-5d %4d evals  %5.1f min  loss %9.2f -> %8.2f  (%d non-finite probes)\n",
            r.seed, DataFrames.nrow(r.df), r.df.t[end] / 60,
            ok[1], minimum(ok), DataFrames.nrow(r.df) - length(ok))
end

# Line-search probes into the unstable region give a non-finite loss and a
# parameter vector that was never accepted. Dropping them keeps the y-limits
# set by the points the optimiser actually moved through.
finite_rows(df) = df[isfinite.(df.loss), :]

"""
Rows where the loss reached a new minimum — the incumbent path.

Every objective evaluation is plotted, but most of them are trial points the
line search rejected; a single probe at mortality = 87 sets that panel's limits
and flattens the actual trajectory to a line. The incumbent is the sequence of
points the optimiser accepted, which is what "the learned parameters over the
run" means, and it is what the y-limits are taken from.
"""
function incumbent_rows(df)
    d = finite_rows(df)
    best = Inf
    keep = falses(DataFrames.nrow(d))
    for i in 1:DataFrames.nrow(d)
        if d.loss[i] < best
            best = d.loss[i]
            keep[i] = true
        end
    end
    return d[keep, :]
end

const COLS = [:steelblue, :orangered, :seagreen, :purple, :goldenrod,
              :crimson, :teal, :darkorange, :magenta, :olivedrab,
              :dodgerblue, :sienna, :mediumvioletred, :darkcyan, :chocolate,
              :royalblue, :forestgreen, :indianred, :slateblue, :darkkhaki]

# Past ~12 runs a per-seed legend is larger than the panel it sits in, and the
# palette has to wrap anyway. The seed table in the corner carries the mapping
# instead.
const SHOW_LEGEND = length(runs) <= 12

# The faint per-evaluation layer is what makes a 3-run plot readable and what
# makes a 10-run plot a smear. Past a few runs, only the accepted path is drawn.
const SHOW_PROBES = length(runs) <= 4
short = [replace(n, "_coeff" => "", "_rate" => "", "_" => " ") for n in PARAM_NAMES]

plt = plot(layout = (4, 3), size = (1500, 1250), dpi = 150, legend = false,
           link = :none, left_margin = 8Plots.mm, bottom_margin = 5Plots.mm)

const ALL = [finite_rows(r.df) for r in runs]
const INC = [incumbent_rows(r.df) for r in runs]

for (pi, name) in enumerate(PARAM_NAMES)
    sym = Symbol(name)
    # Limits from the accepted path, so a rejected probe cannot set the scale.
    lo, hi = extrema(vcat([d[!, sym] for d in INC]...))
    # The random start is a Uniform[0,1) draw regardless of where the coefficient
    # ends up, so a parameter fitted near 0.01 covers two decades on its way
    # down. A linear axis then spends its whole range on the first few
    # evaluations; switch that panel to log.
    uselog = ALLOW_LOG && lo > 0 && hi / lo > 20
    span = hi - lo
    ylo, yhi = if uselog
        (lo / 1.6, hi * 1.6)
    elseif span > 0
        (lo - 0.08span, hi + 0.08span)
    else
        (lo - 1, hi + 1)
    end
    for (ri, _) in enumerate(runs)
        c = COLS[mod1(ri, length(COLS))]
        SHOW_PROBES && plot!(plt[pi], ALL[ri].x, ALL[ri][!, sym], color = c, lw = 0.7,
                             alpha = 0.18, label = "")
        plot!(plt[pi], INC[ri].x, INC[ri][!, sym], color = c,
              lw = SHOW_PROBES ? 2.0 : 1.5, label = "")
    end
    if SHOW_PROBES
        for r in runs
            r.handover === nothing && continue
            vline!(plt[pi], [r.handover], color = :gray, ls = :dot, lw = 1, label = "")
        end
    end
    plot!(plt[pi], title = uselog ? short[pi] * "  (log)" : short[pi],
          titlefontsize = 11, ylims = (ylo, yhi),
          yscale = uselog ? :log10 : :identity,
          xlabel = pi > 6 ? "objective evaluation" : "", ylabel = "value")
end

for (ri, r) in enumerate(runs)
    c = COLS[mod1(ri, length(COLS))]
    plot!(plt[10], INC[ri].x, INC[ri].loss, color = c,
          lw = SHOW_PROBES ? 2.0 : 1.5,
          label = SHOW_LEGEND ? "seed $(r.seed)" : "")
    SHOW_PROBES && plot!(plt[11], ALL[ri].x, max.(ALL[ri].gnorm, 1e-12), color = c,
                         lw = 0.7, alpha = 0.25, label = "")
    plot!(plt[11], INC[ri].x, max.(INC[ri].gnorm, 1e-12), color = c,
          lw = SHOW_PROBES ? 2.0 : 1.5, label = "")
end
plot!(plt[10], title = "loss (running best)", titlefontsize = 11,
      yscale = ALLOW_LOG ? :log10 : :identity,
      xlabel = "objective evaluation", ylabel = "delta-MSE",
      legend = SHOW_LEGEND ? :topright : :none, legendfontsize = 8)
plot!(plt[11], title = "gradient norm", titlefontsize = 11,
      yscale = ALLOW_LOG ? :log10 : :identity,
      xlabel = "objective evaluation", ylabel = "norm")

txt = "L-BFGS, log-parametrised, box [1e-4, 1e4]\n" *
      (SHOW_PROBES ? "bold = accepted (incumbent) path\nfaint = all trial evaluations\n"
                   : "lines = accepted (incumbent) path\n") *
      "independent y-limits per parameter\n\n"
if SHOW_PROBES
    for (ri, r) in enumerate(runs)
        d = INC[ri]
        global txt *= @sprintf("seed %d\n  %d evals (%d accepted), %.0f min\n  loss %.1f -> %.2f\n  |g| %.3g\n\n",
                               r.seed, DataFrames.nrow(r.df), DataFrames.nrow(d),
                               r.df.t[end] / 60, d.loss[1], d.loss[end], d.gnorm[end])
    end
else
    # Two columns once the list is long, so the table still fits the panel.
    ncol = length(runs) > 16 ? 2 : 1
    nr = cld(length(runs), ncol)
    hdr = @sprintf("%-6s %8s %9s", "seed", "loss", "|grad|")
    global txt *= join(fill(hdr, ncol), "    ") * "\n"
    for i in 1:nr
        line = String[]
        for c in 0:(ncol - 1)
            ri = i + c * nr
            ri > length(runs) && continue
            d = INC[ri]
            push!(line, @sprintf("%-6d %8.2f %9.3g",
                                 runs[ri].seed, d.loss[end], d.gnorm[end]))
        end
        global txt *= join(line, "    ") * "\n"
    end
end
plot!(plt[12], framestyle = :none, showaxis = false, grid = false)
annotate!(plt[12], 0.5, 0.5,
          text(txt, length(runs) > 16 ? 6 : 9, :left, :center))

for ext in ("png", "pdf")
    savefig(plt, joinpath(OUTDIR, "lbfgs_parameter_trajectories.$ext"))
end
println("\nwrote ", joinpath(OUTDIR, "lbfgs_parameter_trajectories.png"))
