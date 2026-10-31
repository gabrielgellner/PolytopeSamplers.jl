using PolytopeSamplers: PolytopeSamplers, ACHR, HitAndRun, Polytope, hr_sample, warmup
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

const methods = (HitAndRun(), ACHR())

@testset "hr_sample" begin
    @testset "argument checks" begin
        w = warmup(pentagon)
        @test_throws ArgumentError hr_sample(Xoshiro(1), w, -1)
        @test_throws ArgumentError hr_sample(Xoshiro(1), w, 10; burn_in = -1)
        @test_throws ArgumentError hr_sample(Xoshiro(1), w, 10; thin = 0)
        @test size(hr_sample(Xoshiro(1), w, 0)) == (2, 0)
    end

    @testset "reproducible from a seeded rng ($m)" for m in methods
        w = warmup(pentagon)
        @test hr_sample(Xoshiro(3), w, 100; method = m) == hr_sample(Xoshiro(3), w, 100; method = m)
        @test hr_sample(Xoshiro(3), pentagon, 100; method = m) == hr_sample(Xoshiro(3), w, 100; method = m)
        @test size(hr_sample(w, 7; method = m)) == (2, 7)
    end

    @testset "every point is feasible ($m)" for m in methods
        # The pentagon lifted to three variables with x3 = 6 - x1, x3 ≥ 1 and x2 fixed in a
        # second copy, so every kind of constraint is active
        P = Polytope(
            [0.0, 0.0, 1.0, 2.0], [5.5, 5.5, Inf, 2.0];
            A = [M zeros(3, 2)], b = f, C = [1.0 0.0 1.0 0.0], d = [6.0],
        )
        X = hr_sample(Xoshiro(4), P, 20_000; method = m, burn_in = 100)
        @test all(x -> PolytopeSamplers.isfeasible(P, x; atol = 1.0e-10), eachcol(X))
        @test all(==(2.0), X[4, :])
        @test maximum(abs.(X[1, :] .+ X[3, :] .- 6)) < 1.0e-12
    end

    # Hit-and-run points are correlated, so these tolerances are several standard errors of
    # the thinned chains. Each check runs under several seeds.
    @testset "uniform on the unit square ($m)" for m in methods
        w = warmup(Polytope(zeros(2), ones(2)))
        for seed in 1:5
            X = hr_sample(Xoshiro(seed), w, 20_000; method = m, thin = 5)
            @test all(abs.(mean(X; dims = 2) .- 1 / 2) .< 0.01)
            @test all(abs.(var(X; dims = 2) .- 1 / 12) .< 0.005)
        end
    end

    @testset "uniform on the simplex ($m)" for m in methods
        # The marginals of the uniform distribution on the 3-simplex are Beta(1, 3)
        w = warmup(Polytope(zeros(4), ones(4); C = ones(1, 4), d = [1.0]))
        for seed in 1:5
            X = hr_sample(Xoshiro(seed), w, 20_000; method = m, thin = 10)
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
            X = hr_sample(Xoshiro(seed), w, 20_000; method = m, thin = 5)
            @test all(abs.(mean(X; dims = 2) .- μ) .< 0.06)
            @test all(abs.(var(X; dims = 2) ./ σ2 .- 1) .< 0.05)
        end
    end

    @testset "a single point" begin
        X = hr_sample(Xoshiro(1), Polytope([0.2, 0.7], [0.2, 0.7]), 5)
        @test all(==([0.2, 0.7]), eachcol(X))
    end

    @testset "steps are type stable and allocation free ($m)" for m in methods
        c = PolytopeSamplers.Chain(m, warmup(pentagon))
        rng = Xoshiro(1)
        @test_opt PolytopeSamplers.step!(rng, c)
        @test isempty(check_allocs(PolytopeSamplers.step!, (typeof(rng), typeof(c))))
        @test_opt hr_sample(rng, c.warmup, 10; method = m)
    end
end
