using Test
using InverseTuring
using Random
import Statistics
import ForwardDiff

const REPO = normpath(joinpath(@__DIR__, "..", ".."))
const DATA_DIR = joinpath(REPO, "data")
const HAVE_DATA = isdir(DATA_DIR)

HAVE_DATA || @warn "data/ not found; data-dependent tests will be skipped" DATA_DIR

@testset "InverseTuring" begin

    # =======================================================================
    @testset "parameters" begin
        p = RietkerkParams(collect(1.0:9.0))
        @test paramvector(p) == collect(1.0:9.0)
        @test length(PARAM_NAMES) == NPARAMS == 9
        @test RietkerkParams(paramdict(p)) == p
        @test_throws ArgumentError RietkerkParams([1.0, 2.0])

        # field order must match PARAM_NAMES, or CSVs written by either
        # implementation would be silently scrambled
        for (i, name) in enumerate(PARAM_NAMES)
            @test getfield(p, i) == getfield(p, Symbol(name))
        end

        @test SYNTHETIC_TRUTH.surface_water_diffusion_coeff == 8.0
        @test REALDATA_REFERENCE.mortality_rate == 0.6
        @test eltype(randparams(Xoshiro(0))) === Float64
        @test all(0 .<= paramvector(randparams(Xoshiro(0))) .< 1)
    end

    # =======================================================================
    @testset "laplacian" begin
        # A constant field has zero Laplacian everywhere, including boundaries —
        # this is the property replicate padding is there to guarantee.
        u = fill(3.7, 7, 5)
        @test all(abs.(laplacian(u)) .< 1e-14)

        # Interior stencil against a hand-computed value.
        v = zeros(5, 5)
        v[3, 3] = 1.0
        L = laplacian(v)
        @test L[3, 3] == -4.0
        @test L[2, 3] == L[4, 3] == L[3, 2] == L[3, 4] == 1.0
        @test L[2, 2] == 0.0

        # A linear ramp is harmonic inside; at the edges replicate padding makes
        # the outward difference vanish, leaving the one-sided value.
        r = Float64[j for i in 1:4, j in 1:6]
        Lr = laplacian(r)
        @test all(abs.(Lr[:, 2:5]) .< 1e-14)
        @test all(Lr[:, 1] .≈ 1.0)
        @test all(Lr[:, end] .≈ -1.0)

        # The generic (GPU-safe) path must agree with the fast Array path.
        w = randn(Xoshiro(3), 9, 11)
        fast = laplacian(w)
        generic = similar(w)
        InverseTuring.laplacian!(generic, view(w, :, :))
        @test fast ≈ generic

        # Degenerate shapes.
        @test laplacian(reshape([2.0], 1, 1)) ≈ [0.0;;]
        @test size(laplacian(zeros(1, 4))) == (1, 4)

        @test_throws DimensionMismatch InverseTuring.laplacian!(zeros(3, 3), zeros(4, 4))
    end

    # =======================================================================
    @testset "model step" begin
        # Zero biomass and zero water is a fixed point: nothing should move.
        s = SimState(zeros(4, 4), zeros(4, 4), zeros(4, 4))
        step!(s, SYNTHETIC_TRUTH, 0.0, 0.01)
        @test all(iszero, s.biomass)
        @test all(iszero, s.surface_water)

        # Rain with no vegetation accumulates as surface water only.
        s = SimState(zeros(4, 4), zeros(4, 4), zeros(4, 4))
        step!(s, SYNTHETIC_TRUTH, 10.0, 0.1)
        @test all(s.surface_water .≈ 1.0)
        @test all(iszero, s.soil_water)

        # The update is sequential, not simultaneous: soil water responds to the
        # surface water produced by *this* step. Under a simultaneous update it
        # would still be zero after one step.
        s = SimState(zeros(3, 3), zeros(3, 3), fill(1.0, 3, 3))
        step!(s, SYNTHETIC_TRUTH, 10.0, 0.1)
        @test all(s.soil_water .> 0)

        @test_throws DimensionMismatch SimState(zeros(2, 2), zeros(3, 3), zeros(2, 2))
    end

    # =======================================================================
    @testset "simulate_year" begin
        cfg = SimConfig(steps_per_week = 2, year_time_units = 1.0)
        p = SYNTHETIC_TRUTH

        # Water delivered over a year is invariant to steps_per_week: with no
        # vegetation and no losses, surface water should end up at the annual
        # total regardless of the discretisation.
        dry = RietkerkParams(; surface_water_diffusion_coeff = 0.0,
                             soil_water_diffusion_coeff = 0.0, biomass_diffusion_coeff = 0.0,
                             evaporation_rate = 0.0, seepage_rate = 0.0, mortality_rate = 0.0,
                             infiltration_rate = 0.0, plant_uptake_rate = 0.0,
                             water_use_efficiency = 0.0)
        weekly = sinusoidal_weekly_precip(100.0)
        for spw in (1, 3, 7)
            s = SimState(zeros(3, 3), zeros(3, 3), zeros(3, 3))
            simulate_year!(s, dry, weekly, SimConfig(steps_per_week = spw))
            @test Statistics.mean(s.surface_water) ≈ 100.0 rtol = 1e-10
        end

        # year_time_units genuinely changes the trajectory (it is why the
        # synthetic and real-data fits are not interchangeable).
        s1 = SimState(zeros(4, 4), zeros(4, 4), fill(1.0, 4, 4))
        s2 = SimState(zeros(4, 4), zeros(4, 4), fill(1.0, 4, 4))
        simulate_year!(s1, p, weekly, SimConfig(steps_per_week = 1, year_time_units = 1.0))
        simulate_year!(s2, p, weekly, SimConfig(steps_per_week = 1, year_time_units = 1.5))
        @test !(s1.biomass ≈ s2.biomass)

        @test_throws ArgumentError simulate_year!(s1, p, Float64[], cfg)
        @test_throws ArgumentError simulate_year!(s1, p, weekly, SimConfig(steps_per_week = 0))
    end

    # =======================================================================
    @testset "stability limit" begin
        # d*dt <= 1/4 with dt = year_time_units / (52 * steps_per_week).
        @test diffusion_stability_limit(SimConfig(steps_per_week = 2)) == 26.0
        @test diffusion_stability_limit(SimConfig(steps_per_week = 3)) == 39.0
        @test diffusion_stability_limit(SimConfig(steps_per_week = 4)) == 52.0
        # Stretching a year over 1.5 time units enlarges dt and lowers the bound.
        @test diffusion_stability_limit(SimConfig(steps_per_week = 1, year_time_units = 1.5)) ≈ 52 / 6

        cfg = SimConfig(steps_per_week = 2)          # diffusion-only limit 26
        stable = RietkerkParams(; surface_water_diffusion_coeff = 8.0,
                                soil_water_diffusion_coeff = 1.0, biomass_diffusion_coeff = 0.05,
                                evaporation_rate = 0.3, seepage_rate = 0.4, mortality_rate = 0.3,
                                infiltration_rate = 0.1, plant_uptake_rate = 0.15,
                                water_use_efficiency = 0.35)

        # Combined bound: with small biomass, diffusion dominates and the two
        # agree; with large biomass the reaction terms take over entirely.
        rows(B) = (8 * 8.0 + 0.3 + 0.1B,       # surface water: 8d₁ + l₁ + r₂B
                   8 * 1.0 + 0.4 + 0.15B,      # soil water:    8d₂ + l₂ + r₁B
                   8 * 0.05 + 0.3)             # biomass:       8d₃ + l₃
        @test spectral_bound(stable, 0.0) ≈ maximum(rows(0.0)) ≈ 64.3          # diffusion wins
        @test spectral_bound(stable, 1500.0) ≈ maximum(rows(1500.0)) ≈ 233.4   # soil-water reaction wins
        @test max_stable_dt(stable, 1500.0) ≈ 2 / spectral_bound(stable, 1500.0)

        # The reaction term is the binding one for NDVI-scale biomass: a set with
        # negligible diffusion is still unstable at 1500 g/m².
        lowdiff = RietkerkParams(; surface_water_diffusion_coeff = 0.5,
                                 soil_water_diffusion_coeff = 0.5, biomass_diffusion_coeff = 0.05,
                                 evaporation_rate = 1.0, seepage_rate = 0.9, mortality_rate = 1.0,
                                 infiltration_rate = 0.5, plant_uptake_rate = 1.0,
                                 water_use_efficiency = 0.8)
        @test maximum(paramvector(lowdiff)[1:3]) < diffusion_stability_limit(cfg)  # diffusion says fine
        @test stability_ratio(lowdiff, SimConfig(steps_per_week = 3), 1500.0) > 2   # combined says no
        @test !check_stability(lowdiff, SimConfig(steps_per_week = 3), 1500.0; warn = false)

        # Ratio scales as 1/steps_per_week, so refining always fixes it.
        r3 = stability_ratio(lowdiff, SimConfig(steps_per_week = 3), 1500.0)
        r30 = stability_ratio(lowdiff, SimConfig(steps_per_week = 30), 1500.0)
        @test r3 ≈ 10 * r30
        @test check_stability(lowdiff, SimConfig(steps_per_week = 300), 1500.0; warn = false)

        # The bound must actually predict divergence: just above it, blow up;
        # just below, stay bounded.
        #
        # The initial field must be spatially varying. A uniform field has zero
        # Laplacian, so the diffusion term vanishes identically and an unstable
        # coefficient has nothing to amplify — a uniform start would pass this
        # test at any d. Biomass is held at zero so the surface-water equation
        # stays linear and the only thing under test is the diffusion term.
        weekly = sinusoidal_weekly_precip(300.0)
        for (d, want_bounded) in ((20.0, true), (40.0, false))
            p = RietkerkParams(paramvector(stable) |> v -> (v[1] = d; v))
            sw0 = randn(Xoshiro(4), 16, 16)
            s = SimState(copy(sw0), zeros(16, 16), zeros(16, 16))
            simulate_year!(s, p, weekly, cfg)
            bounded = all(isfinite, s.surface_water) &&
                      maximum(abs, s.surface_water) < 1e6
            @test bounded == want_bounded
        end
    end

    # =======================================================================
    @testset "forcing" begin
        w = sinusoidal_weekly_precip(300.0)
        @test length(w) == 52
        @test annual_total(w) ≈ 300.0            # exact by construction
        @test all(w .>= 0)

        flat = sinusoidal_weekly_precip(300.0; amplitude_fraction = 0.0)
        @test all(flat .≈ flat[1])
        @test annual_total(flat) ≈ 300.0

        # Peak week controls where the maximum lands (index 27 == 0-based week 26).
        @test argmax(sinusoidal_weekly_precip(300.0; peak_week = 26.0)) == 27

        s = summer_weekly_precip(300.0)
        @test length(s) == 52
        @test all(s[1:21] .== 0) && all(s[36:end] .== 0)   # dry outside the pulse
        @test argmax(s) in (28, 29)                        # peak mid-summer
        # Documented quirk: normalised with 365/52 days per week, delivered with 7.
        @test annual_total(s) ≈ 300.0 * 7 / (365 / 52) rtol = 1e-12
        @test annual_total(summer_weekly_precip(300.0; days_per_week = 7)) ≈ 300.0

        @test annual_total(uniform_weekly_precip(365.0)) ≈ 52 * 7
        @test weekly_precip_is_complete(uniform_weekly_precip(300.0), 300.0)
        @test !weekly_precip_is_complete(zeros(52), 300.0)
        @test !weekly_precip_is_complete([fill(1.0, 28); zeros(24)], 400.0)  # truncated mid-year
        @test weekly_precip_is_complete(zeros(52), 0.0)   # no annual total to check against
        @test_throws ArgumentError summer_weekly_precip(300.0; start_week = 10, end_week = 5)
    end

    # =======================================================================
    @testset "loss and gradient" begin
        rng = Xoshiro(11)
        grid = (12, 10)
        target = rand(rng, grid...) .* 5
        tr = SiteTrajectory(copy(target), copy(target),
                            [sinusoidal_weekly_precip(200.0) for _ in 1:3],
                            [rand(rng, grid...) .* 5 for _ in 1:3])
        cfg = SimConfig(steps_per_week = 2)
        prob = InverseProblem([tr], cfg; threaded = false)

        @test ntransitions(prob) == 3
        θ = paramvector(randparams(Xoshiro(5)))
        L = loss(θ, prob)
        @test isfinite(L) && L > 0

        # Averaged vs summed loss differ by exactly the transition count.
        summed = InverseProblem([tr], cfg; average = false, threaded = false)
        @test loss(θ, summed) ≈ L * 3

        # Absolute-field loss is a different objective, and larger here because it
        # also charges for the static spatial mean the delta form cancels out.
        absolute = InverseProblem([tr], cfg; delta_loss = false, threaded = false)
        @test loss(θ, absolute) != L
        @test loss(θ, absolute) > L
        @test rethread(absolute, true).delta_loss == false
        @test rethread(prob, true).threaded && !rethread(prob, false).threaded

        # AD against central differences.
        _, g = loss_and_gradient(prob, θ)
        for i in 1:NPARAMS
            h = 1e-6 * max(abs(θ[i]), 1.0)
            up = copy(θ); up[i] += h
            dn = copy(θ); dn[i] -= h
            fd = (loss(up, prob) - loss(dn, prob)) / (2h)
            @test g[i] ≈ fd rtol = 1e-4 atol = 1e-8
        end

        # Chunk width is a performance knob only; it must not change the answer.
        _, g5 = loss_and_gradient(prob, θ; chunk_size = 5)
        _, g9 = loss_and_gradient(prob, θ; chunk_size = 9)
        @test g5 ≈ g9 rtol = 1e-12

        # A perfect fit on noiseless data has zero loss: generate targets with
        # known parameters and check the loss vanishes there.
        p_true = SYNTHETIC_TRUTH
        init = rand(Xoshiro(7), grid...) .* 3
        st = simstate(init)
        prof = sinusoidal_weekly_precip(180.0)
        tgts = map(1:3) do _
            simulate_year!(st, p_true, prof, cfg)
            copy(st.biomass)
        end
        clean = InverseProblem([SiteTrajectory(copy(init), copy(init),
                                               fill(prof, 3), tgts)], cfg; threaded = false)
        @test loss(paramvector(p_true), clean) < 1e-20
    end

    # =======================================================================
    @testset "threading determinism" begin
        rng = Xoshiro(2)
        grid = (8, 8)
        trs = map(1:4) do _
            init = rand(rng, grid...) .* 4
            SiteTrajectory(init, copy(init),
                           [sinusoidal_weekly_precip(150.0) for _ in 1:2],
                           [rand(rng, grid...) .* 4 for _ in 1:2])
        end
        cfg = SimConfig(steps_per_week = 2)
        θ = paramvector(randparams(Xoshiro(9)))
        serial = loss(θ, InverseProblem(trs, cfg; threaded = false))
        par = loss(θ, InverseProblem(trs, cfg; threaded = true))
        # Bitwise equality, not just approximate: the reduction order is fixed.
        @test serial === par
    end

    # =======================================================================
    @testset "Adam" begin
        # Reproduce PyTorch's update by hand for one step from zero state.
        opt = AdamState(3; lr = 0.1, beta1 = 0.9, beta2 = 0.95, eps = 1e-8)
        θ = [1.0, 2.0, 3.0]
        g = [0.5, -1.0, 0.0]
        adam_step!(opt, θ, g)
        # With zero-initialised moments, the first bias-corrected step is
        # lr * sign(g) for any non-zero gradient.
        @test θ[1] ≈ 1.0 - 0.1 rtol = 1e-6
        @test θ[2] ≈ 2.0 + 0.1 rtol = 1e-6
        @test θ[3] == 3.0                     # zero gradient, zero movement

        # Descends a quadratic.
        opt = AdamState(1; lr = 0.1)
        x = [5.0]
        for _ in 1:400
            adam_step!(opt, x, [2 * x[1]])
        end
        @test abs(x[1]) < 1e-3

        # Gradient clipping matches clip_grad_norm_ semantics.
        g = [3.0, 4.0]
        n = clip_global_norm!(g, 1.0)
        @test n ≈ 5.0
        @test sqrt(sum(abs2, g)) ≈ 1.0 rtol = 1e-6
        g2 = [0.3, 0.4]
        @test clip_global_norm!(copy(g2), 10.0) ≈ 0.5
        @test clip_global_norm!(g2, 10.0) ≈ 0.5 && g2 == [0.3, 0.4]   # untouched

        opt = AdamState(1; lr = 1.0)
        decay_lr!(opt, 0.5)
        @test opt.lr == 0.5
        @test_throws DimensionMismatch adam_step!(AdamState(2; lr = 0.1), [1.0], [1.0])
    end

    # =======================================================================
    @testset "training recovers known parameters" begin
        # Fit two coefficients back from noiseless synthetic data. The full
        # nine-parameter problem is non-identifiable from a short series, so the
        # test perturbs a well-conditioned subset and checks the loss collapses.
        rng = Xoshiro(21)
        init = rand(rng, 10, 10) .* 3
        prof = sinusoidal_weekly_precip(190.0)
        cfg = SimConfig(steps_per_week = 2)
        st = simstate(init)
        tgts = map(1:5) do _
            simulate_year!(st, SYNTHETIC_TRUTH, prof, cfg)
            copy(st.biomass)
        end
        prob = InverseProblem([SiteTrajectory(copy(init), copy(init), fill(prof, 5), tgts)],
                              cfg; threaded = false)

        θ0 = paramvector(SYNTHETIC_TRUTH)
        θ0[6] *= 1.5           # mortality_rate
        θ0[9] *= 0.6           # water_use_efficiency
        start_loss = loss(θ0, prob)

        res = train(prob, θ0; cfg = TrainConfig(epochs = 300, learning_rate = 0.01,
                                                lr_decay = 1.0, verbose = false))
        @test res.converged
        @test res.final_loss < start_loss / 100
        @test length(res.loss_history) == 300
        # Adam is not monotone step to step, so compare windowed averages.
        @test Statistics.mean(res.loss_history[1:10]) >
              100 * Statistics.mean(res.loss_history[(end - 9):end])
        @test all(paramvector(res.params) .>= 1e-4)          # clamp respected

        # Snapshots every save_interval epochs (0, 10, ..., 290), plus a final
        # one at 299 because (epochs - 1) is not a multiple of save_interval.
        # This reproduces the Python rule; the reference pickles have 751
        # snapshots for 7500 epochs for exactly this reason.
        @test length(res.parameter_history) == 31
        @test res.parameter_history[1]["epoch"] == 0
        @test res.parameter_history[end]["epoch"] == 299

        # Snapshots are taken after the optimiser step, as in Python: the entry
        # labelled epoch 0 is the result of epoch 0, not its starting point.
        @test res.parameter_history[1]["mortality_rate"] != θ0[6]
        @test paramvector(res.initial_params) == θ0
        # ...and the last snapshot is the returned parameter vector.
        for (i, n) in enumerate(PARAM_NAMES)
            @test res.parameter_history[end][n] == paramvector(res.params)[i]
        end
    end

    # =======================================================================
    @testset "log-space training" begin
        # Same fixture as above. The point of the log parametrisation is that the
        # optimiser moves in relative rather than absolute steps, so the checks
        # are: the default path is untouched, the feasible set is still enforced,
        # snapshots stay in natural units, and the fit still works.
        rng = Xoshiro(21)
        init = rand(rng, 10, 10) .* 3
        prof = sinusoidal_weekly_precip(190.0)
        cfg = SimConfig(steps_per_week = 2)
        st = simstate(init)
        tgts = map(1:5) do _
            simulate_year!(st, SYNTHETIC_TRUTH, prof, cfg)
            copy(st.biomass)
        end
        prob = InverseProblem([SiteTrajectory(copy(init), copy(init), fill(prof, 5), tgts)],
                              cfg; threaded = false)
        θ0 = paramvector(SYNTHETIC_TRUTH)
        θ0[6] *= 1.5
        θ0[9] *= 0.6
        start_loss = loss(θ0, prob)

        base = TrainConfig(epochs = 300, learning_rate = 0.01, lr_decay = 1.0,
                           verbose = false)
        @test !base.logspace                       # raw is still the default
        lg = TrainConfig(base; logspace = true)
        res = train(prob, θ0; cfg = lg)

        @test res.converged
        @test res.final_loss < start_loss
        @test length(res.loss_history) == 300
        # Positivity is structural under exp, not enforced by the clamp.
        θf = paramvector(res.params)
        @test all(θf .> 0)
        @test all(1e-4 .<= θf .<= 1e4)
        # Snapshots are natural-space parameters, not logs.
        @test length(res.parameter_history) == 31
        for (i, n) in enumerate(PARAM_NAMES)
            @test res.parameter_history[end][n] == θf[i]
        end
        @test paramvector(res.initial_params) == θ0

        # The two parametrisations must actually differ — if they agreed the flag
        # would be doing nothing.
        raw = train(prob, θ0; cfg = base)
        @test paramvector(raw.params) != θf

        # The callback fires on the snapshot cadence with natural-space
        # parameters, so a caller streaming to disk writes exactly the rows
        # `parameter_history` ends up holding.
        seen = Tuple{Int,Vector{Float64},Float64,Float64}[]
        rc = train(prob, θ0; cfg = TrainConfig(lg; epochs = 40, save_interval = 10),
                   callback = (ep, θ, l, g) -> push!(seen, (ep, copy(θ), l, g)))
        @test [s[1] for s in seen] == [0, 10, 20, 30]
        @test length(rc.parameter_history) == 5        # 0,10,20,30 plus final 39
        for (k, s) in enumerate(seen)
            for (i, n) in enumerate(PARAM_NAMES)
                @test rc.parameter_history[k][n] == s[2][i]
            end
            @test s[3] == rc.loss_history[s[1] + 1]
            @test s[4] == rc.gradnorm_history[s[1] + 1]
            @test all(s[2] .> 0)                       # natural units, not logs
        end

        # A start outside the box is projected in before the log is taken, so no
        # domain error and no infinite initial coordinate.
        θbad = copy(θ0)
        θbad[1] = 1e-9
        r2 = train(prob, θbad; cfg = TrainConfig(lg; epochs = 5))
        @test all(isfinite, paramvector(r2.params))
        @test all(paramvector(r2.params) .>= 1e-4)
    end

    # =======================================================================
    @testset "train_many and config" begin
        rng = Xoshiro(31)
        init = rand(rng, 8, 8) .* 3
        prof = sinusoidal_weekly_precip(180.0)
        cfg = SimConfig(steps_per_week = 2)
        st = simstate(init)
        tgts = map(1:2) do _
            simulate_year!(st, SYNTHETIC_TRUTH, prof, cfg)
            copy(st.biomass)
        end
        prob = InverseProblem([SiteTrajectory(copy(init), copy(init), fill(prof, 2), tgts)], cfg)

        tc = TrainConfig(epochs = 20, learning_rate = 0.01, verbose = false)
        seeds = [1, 2, 3]
        # Results must come back in seed order and be independent of where the
        # threads went.
        a = train_many(prob, seeds; cfg = tc, parallel = :none)
        b = train_many(prob, seeds; cfg = tc, parallel = :runs)
        @test length(a) == length(b) == 3
        @test [r.seed for r in a] == seeds
        @test [r.seed for r in b] == seeds
        for (x, y) in zip(a, b)
            @test paramvector(x.params) ≈ paramvector(y.params) rtol = 1e-12
            @test x.final_loss ≈ y.final_loss rtol = 1e-12
        end
        # Same seed => same random start.
        @test paramvector(a[1].initial_params) == paramvector(train(prob; cfg = tc, seed = 1).initial_params)
        @test_throws ArgumentError train_many(prob, seeds; cfg = tc, parallel = :bogus)

        # Copy constructor replaces only the named fields.
        tc2 = TrainConfig(tc; epochs = 99)
        @test tc2.epochs == 99
        @test tc2.learning_rate == tc.learning_rate && tc2.verbose == tc.verbose

        # Chunk width is a performance knob; autotune returns one of its candidates.
        best, timings = autotune_chunk(prob, paramvector(SYNTHETIC_TRUTH); candidates = (3, 9))
        @test best in (3, 9)
        @test Set(keys(timings)) == Set([3, 9])
    end

    # =======================================================================
    @testset "analysis" begin
        # Turing value formula against a hand-evaluated case.
        p = RietkerkParams(; surface_water_diffusion_coeff = 8.0,
                           soil_water_diffusion_coeff = 1.0, biomass_diffusion_coeff = 0.05,
                           evaporation_rate = 0.3, seepage_rate = 0.4, mortality_rate = 0.6,
                           infiltration_rate = 2.1, plant_uptake_rate = 1.9,
                           water_use_efficiency = 0.55)
        d1, d3, l1, l2, l3, r1, r2 = 8.0, 0.05, 0.3, 0.4, 0.6, 1.9, 2.1
        expected = -d1 * l2 * l3 + d3 * (l1 + sqrt(l1 * l2 * r2 / r1)) * (l2 + sqrt(l1 * l2 * r1 / r2))
        @test turing_value(p) ≈ expected

        # Large biomass saturates the composite towards the water-use efficiency.
        @test composite_value(p, 1e6) ≈ p.water_use_efficiency rtol = 1e-4
        @test composite_value(p, 0.0) == 0.0

        # tier1: short history and a floor-pinned parameter are both dropped.
        good = [Dict{String,Float64}(n => 0.5 for n in PARAM_NAMES) for _ in 1:100]
        short = good[1:50]
        degen = deepcopy(good)
        degen[end]["mortality_rate"] = 1e-5
        kept, dropped = tier1_filter(Dict(1 => good, 2 => short, 3 => degen))
        @test kept == [1]
        @test Set(first.(dropped)) == Set([2, 3])
        @test occursin("short history", last(dropped[findfirst(d -> d[1] == 2, dropped)]))
        @test occursin("degenerate", last(dropped[findfirst(d -> d[1] == 3, dropped)]))

        # drop_degenerate is the weaker rule: it ignores history length, so the
        # short-but-healthy run survives where tier1_filter drops it.
        kept2, dropped2 = drop_degenerate(Dict(1 => good, 2 => short, 3 => degen))
        @test kept2 == [1, 2]
        @test first.(dropped2) == [3]
        # Threshold is 0.0011, above the 1e-4 optimiser clamp floor: a parameter
        # sitting on the floor is degenerate under both rules.
        floored = deepcopy(good)
        floored[end]["seepage_rate"] = 1e-4
        @test drop_degenerate(Dict(1 => floored))[1] == Int[]
        @test drop_degenerate(Dict(1 => floored); threshold = 1e-6)[1] == [1]

        # agreement table: identical runs agree perfectly.
        tbl = agreement_table(Dict("mortality_rate" => fill(0.6, 5));
                              ground_truth = REALDATA_REFERENCE)
        @test tbl.cv[1] == 0.0
        @test tbl.agreement_score[1] == 1.0
        @test tbl.mape_vs_gt_pct[1] ≈ 0.0 atol = 1e-12

        # bifurcation sweep: more rain must not give less vegetation, and the
        # requested levels must be the ones snapshotted.
        init = fill(5.0, 12, 12) .+ randn(Xoshiro(8), 12, 12)
        cfg = SimConfig(steps_per_week = 2, year_time_units = 1.0)
        levels = [150.0, 250.0, 350.0]
        means, snaps = bifurcation_sweep(SYNTHETIC_TRUTH, init, levels;
                                         years = 5, cfg = cfg, snapshot_at = [250.0])
        @test length(means) == 3
        @test all(isfinite, means)
        @test issorted(means)
        @test collect(keys(snaps)) == [250.0]
        @test size(snaps[250.0]) == size(init)
        @test Statistics.mean(snaps[250.0]) ≈ means[2]

        # Threading must not change the answer.
        means2, _ = bifurcation_sweep(SYNTHETIC_TRUTH, init, levels; years = 5, cfg = cfg)
        @test means == means2

        # A custom forcing is honoured: constant rain differs from the summer pulse.
        flat, _ = bifurcation_sweep(SYNTHETIC_TRUTH, init, [250.0]; years = 5, cfg = cfg,
                                    forcing = a -> uniform_weekly_precip(a))
        @test flat[1] != means[2]
    end

    # =======================================================================
    @testset "IO round-trip" begin
        dir = mktempdir()
        res = TrainResult(SYNTHETIC_TRUTH, REALDATA_REFERENCE, [3.0, 2.0, 1.0],
                          [Dict("epoch" => 0.0, "mortality_rate" => 0.3)], 3, 1.0, 0.5, true, 42)
        path = save_run(joinpath(dir, "run_00.json"), res; metadata = Dict("run_id" => 7))
        back = load_run(path)
        @test back["final_loss"] == 1.0
        @test back["run_id"] == 7
        @test back["final_params"]["mortality_rate"] == 0.3
        @test back["seed"] == 42

        csv = write_parameter_table(joinpath(dir, "params.csv"), [0, 1],
                                    [SYNTHETIC_TRUTH, REALDATA_REFERENCE];
                                    extra = Dict("turing_value" => [1.0, 2.0]))
        df = read_parameter_table(csv)
        @test names(df)[1] == "model_id"
        @test params_from_row(df[2, :]) == REALDATA_REFERENCE
        @test df.turing_value == [1.0, 2.0]
    end

    # =======================================================================
    @testset "real data" begin
        if !HAVE_DATA
            @test_skip "data directory not available"
        else
            site = load_site(DATA_DIR, "b"; multiplier = 1500.0, T = Float32)
            @test site.name == "subsite_b"
            @test length(site) == 10
            @test years(site) == collect(2013:2022)
            @test size(site) == (131, 140)        # rasterio orientation, not GDAL's

            for o in site.observations
                @test all(0 .<= o.biomass .<= 1500)
                @test length(o.weekly_precipitation) == 52
                @test o.precipitation > 0
            end

            # Same raster read at two precisions must agree to Float32 accuracy.
            f = joinpath(DATA_DIR, "subsite_b", "subsite_b_ndvi", "subsite_b_2013.tif")
            @test ndvi_biomass(f; T = Float32) ≈ Float32.(ndvi_biomass(f; T = Float64)) rtol = 1e-5

            @test_throws ArgumentError ndvi_biomass(joinpath(DATA_DIR, "nope.tif"))
            @test_throws ArgumentError load_site(DATA_DIR, "zzz")

            # Incomplete weekly records fall back to the flat annual total: every year of
            # subsite_k, and subsite_j from 2020 on (the download stops in July 2020).
            k = load_site(DATA_DIR, "k"; T = Float32)
            @test all(o -> o.weekly_precipitation == uniform_weekly_precip(o.precipitation),
                      k.observations)
            j = load_site(DATA_DIR, "j"; T = Float32)
            for o in j.observations
                flat = o.weekly_precipitation == uniform_weekly_precip(o.precipitation)
                @test flat == (o.year >= 2020)
            end
            @test all(o -> all(>=(0), o.weekly_precipitation), site.observations)
            raw = load_site(DATA_DIR, "k"; T = Float32, check_weekly = false)
            @test all(o -> annual_total(o.weekly_precipitation) == 0, raw.observations)

            # Trajectory construction pairs year k's rain with year k+1's biomass.
            tr = SiteTrajectory(site)
            @test length(tr.targets) == 9
            @test tr.targets[1] == site.observations[2].biomass
            @test tr.forcings[1] === site.observations[1].weekly_precipitation
        end
    end

    # =======================================================================
    @testset "PyTorch reference agreement" begin
        metrics = joinpath(REPO, "results", "real_data", "test_results", "test_metrics.csv")
        params = joinpath(REPO, "results", "parameter_history_analysis",
                          "four_site_final_parameter_values.csv")
        if !(HAVE_DATA && isfile(metrics) && isfile(params))
            @test_skip "PyTorch reference outputs not available"
        else
            import CSV, DataFrames
            ref = CSV.read(metrics, DataFrames.DataFrame)
            ptab = read_parameter_table(params)
            cfg = SimConfig(steps_per_week = 4, year_time_units = 1.0)

            mid = 2
            p = params_from_row(only(filter(r -> r.model_id == mid, eachrow(ptab))))
            @test turing_value(p) ≈ only(filter(r -> r.model_id == mid,
                                                eachrow(ptab))).turing_value rtol = 1e-10

            # 1872 explicit Euler steps in Float32, scored exactly as
            # realdata_test_invPDE.py does. Agreement here means the whole
            # pipeline matches, not just individual pieces.
            for sitename in ("subsite_f", "subsite_k")
                # The reference was computed from the raw weekly CSVs.
                series = load_site(DATA_DIR, replace(sitename, "subsite_" => "");
                                   multiplier = 1500.0, T = Float32, check_weekly = false)
                got = evaluate(p, series, cfg)
                want = only(filter(r -> r.model_id == mid && r.site == sitename, eachrow(ref)))
                @test got.num_transitions == want.num_transitions
                @test got.mse ≈ want.mse rtol = 1e-5
                @test got.mae ≈ want.mae rtol = 1e-5
                @test got.correlation ≈ want.correlation rtol = 1e-4
            end
        end
    end

    # =======================================================================
    @testset "Python pickle interop" begin
        pkl = joinpath(REPO, "results", "real_data", "models", "parameters", "model_02_params.pkl")
        if !isfile(pkl)
            @test_skip "Python parameter history not available"
        else
            h = load_parameter_history(pkl)
            @test length(h) == 751
            @test h[1]["epoch"] == 0
            @test h[end]["epoch"] == 7499
            @test all(n -> haskey(h[end], n), PARAM_NAMES)

            # The last snapshot must equal the row the Python analysis exported.
            ptab = read_parameter_table(joinpath(REPO, "results", "parameter_history_analysis",
                                                 "four_site_final_parameter_values.csv"))
            row = only(filter(r -> r.model_id == 2, eachrow(ptab)))
            for n in PARAM_NAMES
                @test h[end][n] ≈ row[n] rtol = 1e-12
            end
        end
    end
end
