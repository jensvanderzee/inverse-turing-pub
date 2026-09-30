"""
    sinusoidal_weekly_precip(annual_total; peak_week = 26.0, amplitude_fraction = 0.7,
                             num_weeks = 52) -> Vector{Float64}

Smooth seasonal forcing: a cosine riding on a constant baseline, rescaled so the
year integrates to `annual_total` mm.

Port of `generate_weekly_precipitation` from the synthetic experiments. The
returned values are rates in mm/day, one per week; `amplitude_fraction = 0`
gives uniform rainfall (used to spin the model up to equilibrium).

The rescaling uses 7 days/week, consistently with [`simulate_year!`](@ref), so
the delivered annual total is exactly `annual_total`.
"""
function sinusoidal_weekly_precip(annual_total::Real; peak_week::Real = 26.0,
                                  amplitude_fraction::Real = 0.7, num_weeks::Integer = 52)
    mean_rate = annual_total / 365
    # Week index is 0-based here to match the Python `range(num_weeks)`.
    raw = [mean_rate * (1 + amplitude_fraction * cospi(2 * (w - peak_week) / num_weeks))
           for w in 0:(num_weeks - 1)]
    raw_total = sum(r * 7 for r in raw)
    scale = raw_total > 0 ? annual_total / raw_total : 1.0
    return raw .* scale
end

"""
    summer_weekly_precip(annual_mm; start_week = 22, end_week = 35,
                         num_weeks = 52) -> Vector{Float64}

Monsoonal forcing: all rain falls in a half-sine pulse over the summer weeks,
the rest of the year is dry. Port of `make_weekly_precip` from
`bifurcation_parallel.py` / `realdata_simulate_invPDE.py`, which is what the
bifurcation diagram is driven by.

`start_week` and `end_week` are **1-based, inclusive** — weeks 22–35 here are the
same weeks as the 0-based `21`–`34` in the Python source (roughly June–August).

!!! note "Reproduced normalisation quirk"
    The pulse is normalised with `365/52 ≈ 7.019` days per week, whereas
    [`simulate_year!`](@ref) converts weekly rates to volumes with exactly 7
    days per week. The model therefore receives `7/(365/52) ≈ 99.73 %` of the
    requested `annual_mm`. This is inherited from the Python implementation and
    kept deliberately, so that a precipitation value on a Julia bifurcation
    diagram means the same thing as on the published one. Pass
    `days_per_week = 7` to drop the quirk.
"""
function summer_weekly_precip(annual_mm::Real; start_week::Integer = 22, end_week::Integer = 35,
                              num_weeks::Integer = 52, days_per_week::Real = 365 / 52)
    1 <= start_week <= end_week <= num_weeks ||
        throw(ArgumentError("need 1 <= start_week <= end_week <= num_weeks"))
    rates = zeros(Float64, num_weeks)
    n_rainy = end_week - start_week + 1
    n_rainy > 1 || throw(ArgumentError("need at least two rainy weeks"))
    for (i, w) in enumerate(start_week:end_week)
        rates[w] = sinpi((i - 1) / (n_rainy - 1))
    end
    shape_sum = sum(rates)
    return rates .* (annual_mm / (shape_sum * days_per_week))
end

"""
    uniform_weekly_precip(annual_mm; num_weeks = 52)

Flat forcing at `annual_mm / 365` mm/day. This is the fallback the Python loader
uses when a site has no weekly precipitation CSV.
"""
uniform_weekly_precip(annual_mm::Real; num_weeks::Integer = 52) =
    fill(annual_mm / 365, num_weeks)

"""
    annual_total(weekly_precip; days_per_week = 7)

Water delivered over a year by a weekly-rate profile, in mm — the quantity the
integrator actually sees.
"""
annual_total(weekly_precip::AbstractVector; days_per_week::Real = 7) =
    sum(weekly_precip) * days_per_week
