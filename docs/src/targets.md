# Non-uniform targets

[`Target`](@ref) is the distribution on a polytope with density proportional to
`exp(ℓ(x))`. `ℓ` is a function of the vector `x`, or any object implementing
[LogDensityProblems](https://github.com/tpapp/LogDensityProblems.jl).

## A Dirichlet distribution on the simplex

The Dirichlet distribution with parameters `α` has log density `Σ (αᵢ - 1) log xᵢ` on the
simplex, and Beta marginals with means `αᵢ / Σα`:

```@example targets
using PolytopeSamplers, Random

α = [2.0, 3.0, 4.0]
P = Polytope(zeros(3), ones(3); C = ones(1, 3), d = [1.0])
t = Target(P, x -> sum((α .- 1) .* log.(x)); gradient = (g, x) -> (g .= (α .- 1) ./ x))

r = sample_until_converged(Xoshiro(1), t, CoordinateHitAndRun(); target_ess = 2_000)
r.diagnostics.mean, α ./ sum(α)
```

## Which samplers support densities

- [`HitAndRun`](@ref) and [`CoordinateHitAndRun`](@ref) draw each point on a chord from the
  density along it by slice sampling with shrinkage (Neal 2003), starting from the whole
  chord. This is exact for any `ℓ` and needs no tuning; it calls `ℓ` a few times per move.
- [`DikinWalk`](@ref) and [`VaidyaWalk`](@ref) add `ℓ(z') - ℓ(z)` to their Metropolis ratio.
- [`CRHMC`](@ref) adds `-ℓ` to its potential and needs the gradient of `ℓ`: pass
  `gradient = (g, x) -> ...`, which fills `g` with `∇ℓ(x)`, or a LogDensityProblems object
  of order 1, for example one built with LogDensityProblemsAD.jl.
- [`BilliardWalk`](@ref) and [`ACHR`](@ref) sample only the uniform distribution and throw
  an `ArgumentError` for a target.

The default method of [`sample_until_converged`](@ref) for a target is
[`HitAndRun`](@ref); [`CoordinateHitAndRun`](@ref) was faster on the targets tested.

## A LogDensityProblems object

```@example targets
using LogDensityProblems

struct Gaussian
    μ::Vector{Float64}
end
LogDensityProblems.dimension(g::Gaussian) = length(g.μ)
LogDensityProblems.capabilities(::Type{Gaussian}) = LogDensityProblems.LogDensityOrder{1}()
LogDensityProblems.logdensity(g::Gaussian, x) = -sum(abs2, x .- g.μ) / 2
LogDensityProblems.logdensity_and_gradient(g::Gaussian, x) = (-sum(abs2, x .- g.μ) / 2, g.μ .- x)

box = Polytope(fill(-1.0, 2), fill(1.0, 2))
r = sample_until_converged(Xoshiro(2), Target(box, Gaussian([0.5, 0.0])), CRHMC(); target_ess = 1_000)
r.diagnostics.mean, r.diagnostics.mcse_mean
```

Truncated to `[-1, 1]`, the exact means are `0.5 - (φ(0.5) - φ(-1.5)) / (Φ(0.5) - Φ(-1.5)) ≈ 0.144`
and 0, for the standard normal density `φ` and distribution function `Φ`.

## Calibration

Over 20 seeds, on a Dirichlet distribution on a simplex, truncated exponentials on a box
with rates 1, 5 and 20, and a Gaussian with correlation 0.8, every supported sampler had
z-scores of its means and standard deviations against the exact values with root mean
square between 0.73 and 1.15.

`ℓ` is called from every chain, on separate threads when Julia has them, so it must be
safe to call concurrently.
