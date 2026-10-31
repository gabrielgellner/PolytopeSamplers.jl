# PolytopeSamplers.jl

Uniform sampling of convex polytopes by Markov chain Monte Carlo: the billiard walk,
coordinate hit-and-run with rounding (CHRR), hit-and-run and artificial centering hit-and-run
(ACHR). Linear programs for the warmup are solved with
[HiGHS](https://highs.dev). The package was called HRMCMC.jl until 2026.

```julia
using Pkg
Pkg.add(url = "https://github.com/gabrielgellner/PolytopeSamplers.jl")
```

## Usage

A `Polytope` is the set of `x` with `lower .<= x .<= upper`, `A * x .<= b` and `C * x .== d`.
The simplest use samples until the chains have converged:

```julia
using PolytopeSamplers, Random

# The triangle x ≥ 0, x1 + x2 ≤ 1
P = Polytope([0.0, 0.0], [1.0, 1.0]; A = [1.0 1.0], b = [1.0])

r = mmcs(Xoshiro(1), P; target_ess = 1_000)
r.converged, r.n                      # converged after r.n points per chain
r.diagnostics.mean, r.diagnostics.mcse_mean
X = reduce(hcat, r.chains)            # one point per column
```

`mmcs` is multiphase Monte Carlo sampling (Chalkis et al. 2021): one linear program finds
the largest ball in the polytope, billiard-walk samples round it, and four billiard-walk
chains run, doubling in length, until the smallest effective sample size reaches
`target_ess` and R̂ is below 1.01. Rounding from samples avoids the dense `m × m`
factorizations of ellipsoid rounding, but up to 2000 constraints it was the slower of the
two (121 s against 10 s with 2000 constraints in 400 dimensions), so
`sample_until_converged(rng, warmup(P; extremes = false))` is the faster route there.

The steps can also be run separately:

```julia
w = warmup(P)                                            # 2n LPs, ellipsoid rounding
r = sample_until_converged(Xoshiro(1), w, BilliardWalk(); target_ess = 1_000)

X = sample(Xoshiro(1), w, BilliardWalk(), 10_000; discard_initial = 1_000)
chains = sample(Xoshiro(1), w, BilliardWalk(), MCMCThreads(), 2_000, 4)
d = diagnostics(chains)
d.min_ess, d.max_rhat     # aim for max_rhat < 1.01
```

`warmup` prepares the polytope:

- `extremes = true` (default) solves `2n` linear programs for the extremes of each
  coordinate, and finds every equality that holds over the polytope: `C * x .== d`, fixed
  coordinates, and inequalities that hold with equality throughout. `extremes = false`
  solves one linear program for the largest inscribed ball instead and falls back to `2n`
  only when that ball has radius zero, as a polytope with room for a ball has no hidden
  equalities. Use it for large problems: with 1000 constraints in 200 dimensions, warmup and
  sampling to 1000 ESS take 4 s with `extremes = false` and 59 s with `extremes = true`.
- `rounding = :ellipsoid` (default) maps the maximum-volume inscribed ellipsoid to the unit
  ball until the polytope is close to round; `:samples` rounds from billiard-walk samples;
  `:none` skips rounding.

`sample` implements the [AbstractMCMC](https://github.com/TuringLang/AbstractMCMC.jl)
interface; `diagnostics` reports ESS, R̂, means and standard deviations with their Monte
Carlo standard errors.

### Samplers

- `BilliardWalk()` (Gryazina & Polyak 2014, in the accelerated form of Chalkis et al. 2021)
  moves along a random direction for a random length, reflecting off the facets. It needs a
  few steps per effective sample in every dimension tested (10 to 200) and is the
  recommended method.
- `CoordinateHitAndRun()` on the rounded warmup is CHRR (Haraldsdóttir et al. 2017), the
  COBRA Toolbox default. Each step makes one coordinate move per dimension at `O(m)` each,
  in cyclic order by default; moves per effective sample grow as the square of the
  dimension.
- `HitAndRun()` draws directions uniformly from the sphere. `ACHR()` draws them from the
  running mean to a warmup point; it is only approximately Markov, needs
  `warmup(P; extremes = true)`, and is the slowest method on rounded polytopes.

On the 63-dimensional food web polytope of the example (four chains of 2000 points):

| Sampler | min ESS | max R̂ | ESS/s |
|---|---|---|---|
| HitAndRun, unrounded (the 2019 version) | 19 | 2.03 | 17 |
| CoordinateHitAndRun | 1464 | 1.005 | 2210 |
| BilliardWalk | 7262 | 1.001 | 6690 |

Estimates agree with hopsy 1.7 on the same polytopes within their Monte Carlo standard
errors.

### Threads

Start Julia with threads (`julia --threads=auto`) and two kinds of work run in parallel:

- Chains: `sample_until_converged`, `mmcs` and `sample(..., MCMCThreads(), n, nchains)`
  run each chain on its own thread. Each chain has its own generator seeded from `rng`, so
  results are identical for any number of threads, and billiard chains share one copy of
  `G Gᵀ`.
- Independent problems: `warmup` builds its own JuMP model and every function takes an
  explicit `rng`, so `Threads.@threads` over polytopes is safe;
  `examples/foodweb_replicates.jl` runs one food web per thread.

Set `BLAS.set_num_threads(1)` when Julia threads keep every core busy. Measured on a
10-core Apple machine (4 performance and 6 efficiency cores):

| Work | 1 thread | 4 threads | 10 threads |
|---|---|---|---|
| 80 food web replicates (63 dimensions, to 1000 ESS each) | 42.0 s | 17.9 s | 10.4 s |
| 4 billiard chains of 4000 steps, 200 dimensions | 5.3 s | 1.6 s | |

Chains in one run gain little past the number of performance cores, as the slowest chain
sets the time; batches of independent problems keep gaining.

### Other details

- `isempty(P)` checks feasibility with one linear program. `warmup` throws an
  `ArgumentError` for empty or unbounded polytopes.
- Every function takes an optional leading `rng`. Chains have their own generators seeded
  from it, so results do not depend on the number of threads. Each `warmup` builds its own
  JuMP model, so separate polytopes can be set up in parallel tasks. Pass `optimizer` to use
  a different JuMP solver.

`examples/constrained_foodweb.jl` samples the interaction strengths of a food web consistent
with a given equilibrium, the problem of
[constrained-foodwebs](https://github.com/gabrielgellner/constrained-foodwebs), with webs from
[FoodWebs.jl](https://github.com/gabrielgellner/FoodWebs.jl). Run it with
`just example constrained_foodweb`; `just example foodweb_replicates` runs 80 webs in
parallel.

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
just ci         # formatting check, JETLS lint and tests; run before every push and merge
just stress     # convergence stress tests and mixing-cost scaling (about 3 minutes)
```

## License

GPL-3.0-or-later, see `LICENSE.md`. Parts of the package adapt code from other open-source
samplers; each such file names its source and license.
