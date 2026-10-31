using PolytopeSamplers: PolytopeSamplers, ACHR, BilliardWalk, CRHMC, CoordinateHitAndRun,
    DikinWalk, HitAndRun, Polytope, Target, VaidyaWalk, sample, sample_until_converged, warmup
using LogDensityProblems: LogDensityProblems, LogDensityOrder
using Random: Xoshiro
using Statistics: cor, mean

const density_methods = (
    HitAndRun(), CoordinateHitAndRun(), CoordinateHitAndRun(scan = :random), DikinWalk(),
    VaidyaWalk(), CRHMC(),
)

# Means within five and standard deviations within six Monte Carlo standard errors
function moments_match(d, μ, σ)
    return all(@. abs(d.mean - μ) < 5 * d.mcse_mean) && all(@. abs(d.std - σ) < 6 * d.mcse_std)
end

# A log density implementing LogDensityProblems directly: an isotropic Gaussian
struct Gaussian
    μ::Vector{Float64}
    σ::Float64
end
LogDensityProblems.dimension(g::Gaussian) = length(g.μ)
LogDensityProblems.capabilities(::Type{Gaussian}) = LogDensityOrder{1}()
LogDensityProblems.logdensity(g::Gaussian, x) = -sum(abs2, x .- g.μ) / (2 * g.σ^2)
function LogDensityProblems.logdensity_and_gradient(g::Gaussian, x)
    return LogDensityProblems.logdensity(g, x), -(x .- g.μ) ./ g.σ^2
end

@testset "non-uniform targets" begin
    @testset "Target" begin
        P = Polytope(zeros(3), ones(3))
        t = Target(P, x -> -sum(x))
        @test t isa Target
        @test sprint(show, t) == "Target on a 3-dimensional slice of a 3-variable polytope"
        @test LogDensityProblems.capabilities(t.logdensity) == LogDensityOrder{0}()
        tg = Target(P, x -> -sum(x); gradient = (g, x) -> fill!(g, -1))
        @test LogDensityProblems.capabilities(tg.logdensity) == LogDensityOrder{1}()
        value, gradient = LogDensityProblems.logdensity_and_gradient(tg.logdensity, [0.1, 0.2, 0.3])
        @test value ≈ -0.6
        @test gradient == fill(-1.0, 3)
        @test Target(warmup(P), Gaussian(zeros(3), 1.0)) isa Target
        @test_throws DimensionMismatch Target(P, Gaussian(zeros(2), 1.0))
        @test_throws ArgumentError Target(P, Gaussian(zeros(3), 1.0); gradient = (g, x) -> g)
        @test_throws ArgumentError Target(P, 1.0)
    end

    @testset "unsupported samplers" begin
        t = Target(Polytope(zeros(3), ones(3)), x -> -sum(x))
        @test_throws ArgumentError sample(Xoshiro(1), t, BilliardWalk(), 10)
        @test_throws ArgumentError sample(Xoshiro(1), t, ACHR(), 10)
        # CRHMC needs the gradient
        @test_throws ArgumentError sample(Xoshiro(1), t, CRHMC(), 10)
    end

    @testset "reproducible ($m)" for m in density_methods
        t = Target(Polytope(zeros(3), ones(3)), x -> -sum(x); gradient = (g, x) -> fill!(g, -1))
        @test sample(Xoshiro(2), t, m, 50) == sample(Xoshiro(2), t, m, 50)
        @test all(x -> x in t.warmup.polytope, eachcol(sample(Xoshiro(2), t, m, 200)))
    end

    @testset "Dirichlet on the simplex ($m)" for m in density_methods
        # Marginals of Dirichlet(α) are Beta(αᵢ, Σα - αᵢ)
        α = [2.0, 3.0, 4.0, 5.0, 6.0]
        a = sum(α)
        P = Polytope(zeros(5), ones(5); C = ones(1, 5), d = [1.0])
        t = Target(P, x -> sum((α .- 1) .* log.(x)); gradient = (g, x) -> (g .= (α .- 1) ./ x))
        r = sample_until_converged(Xoshiro(3), t, m; target_ess = 1_000)
        @test r.converged
        @test moments_match(r.diagnostics, α ./ a, sqrt.(α .* (a .- α) ./ (a^2 * (a + 1))))
    end

    @testset "truncated exponentials on a box ($m)" for m in density_methods
        # exp(-λᵢ xᵢ) on [0, 1]: mean 1/λ - 1/(e^λ - 1), variance 1/λ² - e^λ/(e^λ - 1)², a
        # concentrated density for λ = 20
        λ = [1.0, 5.0, 20.0]
        t = Target(Polytope(zeros(3), ones(3)), x -> -sum(λ .* x); gradient = (g, x) -> (g .= -λ))
        μ = @. 1 / λ - 1 / (exp(λ) - 1)
        σ = @. sqrt(1 / λ^2 - exp(λ) / (exp(λ) - 1)^2)
        r = sample_until_converged(Xoshiro(4), t, m; target_ess = 1_000)
        @test r.converged
        @test moments_match(r.diagnostics, μ, σ)
    end

    @testset "correlated Gaussian ($m)" for m in density_methods
        # N(0, Σ) with correlation 0.8, inside a box 8 standard deviations wide
        Σ = [1.0 0.8; 0.8 1.0]
        Σinv = inv(Σ)
        t = Target(
            Polytope(fill(-8.0, 2), fill(8.0, 2)), x -> -(x' * Σinv * x) / 2;
            gradient = (g, x) -> (g .= -(Σinv * x)),
        )
        r = sample_until_converged(Xoshiro(5), t, m; target_ess = 1_000)
        @test r.converged
        @test moments_match(r.diagnostics, zeros(2), ones(2))
        # The correlation has standard error about (1 - ρ²)/√ESS = 0.011
        @test cor(reduce(hcat, r.chains)')[1, 2] ≈ 0.8 atol = 0.06
    end
end
