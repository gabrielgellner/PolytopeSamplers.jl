# Example: constrained food webs

The package began as the sampler for
[constrained-foodwebs](https://github.com/gabrielgellner/constrained-foodwebs): finding the
interaction strengths of a food web that are consistent with an observed equilibrium.

## The polytope

Lotka–Volterra dynamics `dxᵢ/dt = xᵢ (rᵢ + Σⱼ aᵢⱼ xⱼ)` have an equilibrium at the biomasses
`x` when `rᵢ + Σⱼ aᵢⱼ xⱼ = 0` for every species. For a given web, biomasses and growth rates,
the magnitudes `αᵢⱼ = |aᵢⱼ|` of the links that exist form a polytope:

- one equality per species, `Σⱼ sᵢⱼ xⱼ αᵢⱼ + rᵢ = 0`, with `sᵢⱼ` the sign of the interaction
  (−1 for predation on `i` and for self-limitation, +1 for consumption by `i`);
- bounds `lb ≤ αᵢⱼ ≤ ub`;
- optionally, for each predator `j` of prey `i`, `αᵢⱼ ≥ αⱼᵢ`: a predator gains less from a
  unit of prey than the prey loses.

The repository's `examples/foodweb_polytope.jl` builds this polytope from a
[FoodWebs.jl](https://github.com/gabrielgellner/FoodWebs.jl) web. A 15-species web gives 78
interaction strengths in a 63-dimensional slice.

## Sampling

```julia
include("examples/foodweb_polytope.jl")

rng = Xoshiro(2026)
web = generalized_cascade_network(rng, 15, 0.15; basal_paths = true)
x = pyramid_biomass(rng, web)
r = [i in basal_species(web) ? 5 * rand(rng) : -rand(rng) for i in 1:15]
P, links = foodweb_polytope(web, x, r)

result = sample_until_converged(rng, warmup(P); target_ess = 1_000)
```

Each sample is one community matrix `J = diag(x) A` consistent with the equilibrium; the
fraction of them with all eigenvalues in the left half-plane is the probability that the
equilibrium is stable.

## Results

On this web (four chains of 2000 points, `just example constrained_foodweb`):

| Sampler | min ESS | max R̂ | ESS/s | stable fraction |
|---|---|---|---|---|
| HitAndRun, unrounded (the 2019 version) | 18 | 2.18 | 21 | 0.744 |
| ACHR, unrounded (as in the original study) | 20 | 1.85 | 45 | 0.680 |
| CoordinateHitAndRun | 2867 | 1.002 | 4810 | 0.703 |
| BilliardWalk | 6868 | 1.002 | 7000 | 0.708 |

The methods used before this package had not converged at this length, and their stable
fractions disagree; the converged samplers agree with each other. Results from the
original study were computed with ACHR on single chains without convergence diagnostics
and should be rechecked.

`examples/foodweb_replicates.jl` runs 80 webs to 1000 effective samples each, one per
thread (see [Parallel sampling](@ref)).
