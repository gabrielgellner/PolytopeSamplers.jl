# PolytopeSamplers.jl

Uniform sampling of convex polytopes by Markov chain Monte Carlo: hit-and-run and artificial
centering hit-and-run (ACHR). Linear programs for the warmup are solved with
[HiGHS](https://highs.dev). The package was called HRMCMC.jl until 2026.

```julia
using Pkg
Pkg.add(url = "https://github.com/gabrielgellner/PolytopeSamplers.jl")
```

## Usage

A `Polytope` is the set of `x` with `lower .<= x .<= upper`, `A * x .<= b` and `C * x .== d`.
`warmup` minimizes and maximizes each coordinate by linear programming to find a starting
point and the directions in which the polytope has width. `sample` implements the
[AbstractMCMC](https://github.com/TuringLang/AbstractMCMC.jl) interface: it returns one
point per column, or one matrix per chain when run with `MCMCThreads()`, and `diagnostics`
reports the effective sample size and R̂ of each coordinate.

```julia
using PolytopeSamplers, Random

# The triangle x ≥ 0, x1 + x2 ≤ 1
P = Polytope([0.0, 0.0], [1.0, 1.0]; A = [1.0 1.0], b = [1.0])
w = warmup(P)
X = sample(Xoshiro(1), w, ACHR(), 10_000; discard_initial = 1_000, thinning = 10)

chains = sample(Xoshiro(1), w, HitAndRun(), MCMCThreads(), 2_000, 4; thinning = 10)
d = diagnostics(chains)
d.min_ess, d.max_rhat
```

- `method = HitAndRun()` draws directions uniformly from the sphere; the chain is Markov with
  the uniform distribution as its stationary distribution. `ACHR()` draws directions from the
  running mean of the chain to a warmup point, which mixes faster in long thin polytopes
  but is only approximately Markov.
- Equalities can come from `C * x .== d`, from coordinates with `lower[i] == upper[i]`, or
  from inequalities that hold with equality over the whole polytope. `warmup` finds all
  three and keeps chains on the slice they define.
- `isempty(P)` checks feasibility with one linear program. `warmup` throws an
  `ArgumentError` for empty or unbounded polytopes.
- Every function takes an optional leading `rng`. Chains from `MCMCThreads()` share one
  warmup, and each `warmup` builds its own JuMP model, so separate polytopes can be set up
  in parallel tasks. Pass `optimizer` to use a
  different JuMP solver.

`examples/constrained_foodweb.jl` samples the interaction strengths of a food web consistent
with a given equilibrium, the problem of
[constrained-foodwebs](https://github.com/gabrielgellner/constrained-foodwebs), with webs from
[FoodWebs.jl](https://github.com/gabrielgellner/FoodWebs.jl). Run it with
`just example constrained_foodweb`.

## Changes from the 2019 version (HRMCMC.jl)

- `ACHRSampler(problem)` and `hr_sample!(sampler, n)` are replaced by
  `sample(rng, warmup(P), ACHR(), n)`. The problem `E * α + r = 0, L * α >= 0`
  is `Polytope(lower, upper; C = E, d = -r, A = -L, b = zeros(size(L, 1)))`.
- `n` is now the number of points returned; the chain runs
  `discard_initial + (n - 1) * thinning + 1` steps.
- Plain hit-and-run drew its directions from the positive orthant of the nullspace, as
  `rand` replaced `randn`; it now draws them uniformly from the sphere.
- The step bounded each coordinate by `(1 - tol) * bound`, which loosens negative bounds; it
  now clamps slacks at zero.
- GLPK and Gurobi are replaced by HiGHS, and RecursiveArrayTools, Distributions and
  Parameters are no longer dependencies.

## Development

Tasks are run with [just](https://github.com/casey/just):

```sh
just setup      # install Runic and JETLS as Julia apps, instantiate environments
just fmt        # format with Runic
just check      # formatting check, JETLS lint, and tests (what CI runs)
```

## License

GPL-3.0-or-later, see `LICENSE.md`. Parts of the package adapt code from other open-source
samplers; each such file names its source and license.
