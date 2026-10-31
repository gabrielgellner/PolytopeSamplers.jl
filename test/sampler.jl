using PolytopeSamplers: PolytopeSamplers, ACHR, CoordinateHitAndRun, HitAndRun, MCMCSerial, MCMCThreads, Polytope, sample, warmup
using AllocCheck: check_allocs
using JET: @test_opt
using Random: Xoshiro
using Statistics: mean, var

# A pentagon from the original test of the package, M * x ≤ f in the box [0, 5.5]^2
const M = [3 / 2 -1; 1 1; -2 -1]
const f = [5.0, 10.0, -3.0]
const pentagon = Polytope([0.0, 0.0], [5.5, 5.5]; A = M, b = f)

# Mean and variance of the uniform distribution on the pentagon, by rejection sampling
function pentagon_moments(rng, n)
    X = 5.5 .* rand(rng, 2, n)
    keep = [all(M * x .<= f) for x in eachcol(X)]
    Y = X[:, keep]
    return vec(mean(Y; dims = 2)), vec(var(Y; dims = 2))
end

const methods = (HitAndRun(), ACHR(), CoordinateHitAndRun())

@testset "sample" begin
    @testset "output shape ($m)" for m in methods
        w = warmup(pentagon)
        @test size(sample(Xoshiro(1), w, m, 7)) == (2, 7)
        @test size(sample(w, m, 1)) == (2, 1)
        @test size(sample(pentagon, m, 3)) == (2, 3)
        chains = sample(Xoshiro(1), w, m, MCMCSerial(), 5, 3)
        @test chains isa Vector{Matrix{Float64}}
        @test size.(chains) == fill((2, 5), 3)
        @test_throws ArgumentError sample(Xoshiro(1), w, m, 10; discard_initial = -1)
    end

    @testset "reproducible from a seeded rng ($m)" for m in methods
        w = warmup(pentagon)
        X = sample(Xoshiro(3), w, m, 100)
        @test sample(Xoshiro(3), w, m, 100) == X
        @test sample(Xoshiro(3), pentagon, m, 100) == X
        @test sample(Xoshiro(3), w, m, MCMCThreads(), 50, 4) ==
            sample(Xoshiro(3), pentagon, m, MCMCSerial(), 50, 4)
        # The first point is one step from the warmup center
        @test X[:, 1] != w.center
    end

    @testset "discard_initial and thinning ($m)" for m in methods
        w = warmup(pentagon)
        long = sample(Xoshiro(5), w, m, 30)
        @test sample(Xoshiro(5), w, m, 10; discard_initial = 5) == long[:, 6:15]
        @test sample(Xoshiro(5), w, m, 10; thinning = 3) == long[:, 1:3:28]
    end

    @testset "every point is feasible ($m)" for m in methods
        # The pentagon lifted to three variables with x3 = 6 - x1, x3 ≥ 1 and x2 fixed in a
        # second copy, so every kind of constraint is active
        P = Polytope(
            [0.0, 0.0, 1.0, 2.0], [5.5, 5.5, Inf, 2.0];
            A = [M zeros(3, 2)], b = f, C = [1.0 0.0 1.0 0.0], d = [6.0],
        )
        X = sample(Xoshiro(4), P, m, 20_000; discard_initial = 100)
        @test all(x -> PolytopeSamplers.isfeasible(P, x; atol = 1.0e-10), eachcol(X))
        @test all(==(2.0), X[4, :])
        @test maximum(abs.(X[1, :] .+ X[3, :] .- 6)) < 1.0e-12
    end

    # Hit-and-run points are correlated, so these tolerances are several standard errors of
    # the thinned chains. Each check runs under several seeds.
    @testset "uniform on the unit square ($m)" for m in methods
        w = warmup(Polytope(zeros(2), ones(2)))
        for seed in 1:5
            X = sample(Xoshiro(seed), w, m, 20_000; thinning = 5)
            @test all(abs.(mean(X; dims = 2) .- 1 / 2) .< 0.01)
            @test all(abs.(var(X; dims = 2) .- 1 / 12) .< 0.005)
        end
    end

    @testset "uniform on the simplex ($m)" for m in methods
        # The marginals of the uniform distribution on the 3-simplex are Beta(1, 3)
        w = warmup(Polytope(zeros(4), ones(4); C = ones(1, 4), d = [1.0]))
        for seed in 1:5
            X = sample(Xoshiro(seed), w, m, 20_000; thinning = 10)
            @test all(abs.(mean(X; dims = 2) .- 1 / 4) .< 0.01)
            @test all(abs.(var(X; dims = 2) .- 3 / 80) .< 0.004)
        end
    end

    @testset "uniform on the pentagon ($m)" for m in methods
        μ, σ2 = pentagon_moments(Xoshiro(0), 2_000_000)
        w = warmup(pentagon)
        # The coordinates have standard deviation 1.4, so the means of 20_000 nearly
        # independent points have standard error 0.01
        for seed in 1:5
            X = sample(Xoshiro(seed), w, m, 20_000; thinning = 5)
            @test all(abs.(mean(X; dims = 2) .- μ) .< 0.06)
            @test all(abs.(var(X; dims = 2) ./ σ2 .- 1) .< 0.05)
        end
    end

    @testset "a single point ($m)" for m in methods
        X = sample(Xoshiro(1), Polytope([0.2, 0.7], [0.2, 0.7]), m, 5)
        @test all(==([0.2, 0.7]), eachcol(X))
    end

    @testset "unrounded warmup ($m)" for m in methods
        w = warmup(pentagon; rounding = false)
        X = sample(Xoshiro(1), w, m, 2000)
        @test all(x -> x in pentagon, eachcol(X))
    end

    @testset "CoordinateHitAndRun moves" begin
        @test_throws ArgumentError CoordinateHitAndRun(moves = 0)
        @test sprint(show, CoordinateHitAndRun()) == "CoordinateHitAndRun()"
        @test sprint(show, CoordinateHitAndRun(moves = 3)) == "CoordinateHitAndRun(moves = 3)"
        w = warmup(pentagon)
        # Steps of one move, keeping every second, follow steps of two moves
        X2 = sample(Xoshiro(2), w, CoordinateHitAndRun(moves = 2), 2)
        X1 = sample(Xoshiro(2), w, CoordinateHitAndRun(moves = 1), 2; discard_initial = 1, thinning = 2)
        @test X2 ≈ X1
    end

    @testset "steps are type stable and allocation free ($m)" for m in methods
        c = PolytopeSamplers.HitAndRunState(m, warmup(pentagon))
        rng = Xoshiro(1)
        @test_opt PolytopeSamplers.step!(rng, c)
        @test isempty(check_allocs(PolytopeSamplers.step!, (typeof(rng), typeof(c))))
    end
end
