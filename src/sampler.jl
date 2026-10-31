"""
    PolytopeSampler

Supertype of the samplers: the hit-and-run methods ([`AbstractHitAndRun`](@ref)) and
[`BilliardWalk`](@ref). All of them move in the coordinates of a [`Warmup`](@ref), which are
rounded by default. A sampler `S` implements `chain_state(S, w)`, returning a state with
fields `warmup` and `x` (the current point), and `step!(rng, state)`.
"""
abstract type PolytopeSampler <: AbstractSampler end

"""
    AbstractHitAndRun

Supertype of the hit-and-run methods, which differ in how they choose a direction:
[`HitAndRun`](@ref), [`ACHR`](@ref) and [`CoordinateHitAndRun`](@ref).
"""
abstract type AbstractHitAndRun <: PolytopeSampler end

"""
    HitAndRun()

Hit-and-run with directions drawn uniformly from the unit sphere in the coordinates of the
warmup (Smith 1984). The chain is Markov and its stationary distribution is uniform on the
polytope. Each step costs `O(mk)` for `m` constraints in `k` dimensions.

Smith, R. L. (1984). Efficient Monte Carlo procedures for generating points uniformly
distributed over bounded regions. Operations Research 32, 1296–1308.
<https://doi.org/10.1287/opre.32.6.1296>
"""
struct HitAndRun <: AbstractHitAndRun end

"""
    ACHR()

Artificial centering hit-and-run (Kaufman & Smith 1998), as in the COBRA Toolbox: each
direction runs from the running mean of the chain to a random extreme point of a
coordinate. A warmup built with `extremes = false` (the default) solves the `2n` linear
programs for these points when the first ACHR chain starts, and its chains share them. Directions
follow the shape of the polytope, so long thin polytopes mix faster than with
[`HitAndRun`](@ref) when the warmup is not rounded. The direction depends on the history
through the mean, so the chain is only approximately Markov; its distribution approaches
uniform as the mean settles.

Kaufman, D. E. & Smith, R. L. (1998). Direction choice for accelerated convergence in
hit-and-run sampling. Operations Research 46, 84–95. <https://doi.org/10.1287/opre.46.1.84>
"""
struct ACHR <: AbstractHitAndRun end

"""
    CoordinateHitAndRun(; moves = nothing, scan = :cyclic)

Coordinate hit-and-run: each move resamples one coordinate of the warmup uniformly on its
chord. With `scan = :cyclic` (the default, as in hopsy) the coordinates are taken in turn;
with `scan = :random` each is chosen at random and the chain is reversible. Both keep the
uniform distribution stationary; the cyclic scan gave 1.3 to 1.8 times the effective
samples per move on the polytopes of the test suite. On a rounded warmup this is CHRR (Haraldsdóttir et al.
2017), the default sampler of the COBRA Toolbox. A move costs `O(m)` for `m` constraints,
as the slacks are updated in place; each step of the chain makes `moves` moves (default:
the dimension `k` of the warmup), so a step costs about as much as one of
[`HitAndRun`](@ref). The chain is Markov, with the uniform distribution as its stationary distribution (with
either scan). Coordinate moves per effective sample grow
as `k²`.

A chain started exactly at a vertex never moves when no coordinate axis points into the
polytope there, as on a simplex; chains started by `sample` begin at the interior center.
[`BilliardWalk`](@ref) leaves corners quickly and needs fewer steps per effective sample.

Haraldsdóttir, H. S., Cousins, B., Thiele, I., Fleming, R. M. T. & Vempala, S. (2017).
CHRR: coordinate hit-and-run with rounding for uniform sampling of constraint-based models.
Bioinformatics 33, 1741–1743. <https://doi.org/10.1093/bioinformatics/btx052>
"""
Base.@kwdef struct CoordinateHitAndRun <: AbstractHitAndRun
    moves::Union{Nothing, Int} = nothing
    scan::Symbol = :cyclic
    function CoordinateHitAndRun(moves, scan)
        moves === nothing || moves >= 1 || throw(ArgumentError("moves must be positive, got $moves"))
        scan in (:random, :cyclic) || throw(ArgumentError("scan must be :random or :cyclic, got :$scan"))
        return new(moves, scan)
    end
end

function Base.show(io::IO, m::CoordinateHitAndRun)
    args = String[]
    m.moves === nothing || push!(args, "moves = $(m.moves)")
    m.scan === :cyclic || push!(args, "scan = :$(m.scan)")
    print(io, "CoordinateHitAndRun(", join(args, ", "), ")")
    return
end

# Coefficients below this are treated as zero: a direction component this small moves a
# constraint by rounding error only.
const ZERO_TOL = 1.0e-12

# Samplers update the slacks `h - G z` after each move and recompute them from `z` every
# this many steps, so that rounding error does not accumulate (hops updates them in place
# without recomputing)
const SLACK_REFRESH = 100

# The state of one chain, in the coordinates `z` of the warmup, with buffers so that a step
# does not allocate
mutable struct HitAndRunState{M <: AbstractHitAndRun}
    const method::M
    const warmup::Warmup
    const z::Vector{Float64}
    const x::Vector{Float64} # center + transform * z
    const u::Vector{Float64} # direction
    const slack::Vector{Float64} # h - G z
    const Gu::Vector{Float64}
    const center::Vector{Float64} # running mean of z, for ACHR
    const zpoints::Matrix{Float64} # extreme points in z, for ACHR
    n_seen::Int # points in the running mean
    steps::Int # steps taken, for refreshing the slacks
    next_coordinate::Int # for the cyclic scan
end

chain_state(method::AbstractHitAndRun, w::Warmup) = HitAndRunState(method, w)

function HitAndRunState(method::AbstractHitAndRun, w::Warmup)
    k = size(w.transform, 2)
    m = length(w.h)
    zpoints = method isa ACHR ? extreme_zpoints(w) : w.zpoints
    center = isempty(zpoints) ? zeros(k) : vec(mean(zpoints; dims = 2))
    return HitAndRunState(
        method, w, zeros(k), copy(w.center), zeros(k), copy(w.h), zeros(m), center, zpoints,
        size(zpoints, 2), 0, 1,
    )
end

# Set `c.u` to a unit direction; return false when there is none, as for a single point
function direction!(rng::AbstractRNG, c::HitAndRunState{HitAndRun})
    isempty(c.u) && return false
    randn!(rng, c.u)
    normalize!(c.u)
    return true
end

function direction!(rng::AbstractRNG, c::HitAndRunState{ACHR})
    isempty(c.u) && return false
    zp = c.zpoints
    j = rand(rng, axes(zp, 2))
    for i in eachindex(c.u)
        c.u[i] = zp[i, j] - c.center[i]
    end
    r = norm(c.u)
    r > ZERO_TOL || return false
    c.u ./= r
    return true
end

# Narrow `(tmin, tmax)` to the `t` with `t * g <= s`, for slack `s >= 0`
@inline function narrow(tmin, tmax, g, s)
    if g > ZERO_TOL
        tmax = min(tmax, s / g)
    elseif g < -ZERO_TOL
        tmin = max(tmin, s / g)
    end
    return tmin, tmax
end

# The interval of `t` with `G (z + t g) ≤ h`, given the slacks `h - G z` and the rates
# `g = G u`. Slacks are clamped at zero so that a point that rounding has put just outside a
# constraint still gets an interval holding zero.
function chord(slack, g)
    tmin, tmax = -Inf, Inf
    for i in eachindex(slack, g)
        tmin, tmax = narrow(tmin, tmax, g[i], max(slack[i], 0.0))
    end
    isfinite(tmin) && isfinite(tmax) ||
        throw(ArgumentError("the polytope is unbounded along a sampled direction"))
    return tmin, tmax
end

uniform(rng, tmin, tmax) = tmin + (tmax - tmin) * rand(rng)

# One step: move `c.z` along a new direction to a uniform point on the chord, update the
# running mean and `c.x`
function step!(rng::AbstractRNG, c::HitAndRunState)
    w = c.warmup
    if direction!(rng, c)
        refresh_slack!(c)
        mul!(c.Gu, w.G, c.u)
        t = uniform(rng, chord(c.slack, c.Gu)...)
        for i in eachindex(c.z)
            c.z[i] += t * c.u[i]
        end
        for i in eachindex(c.slack)
            c.slack[i] -= t * c.Gu[i]
        end
    end
    return finish_step!(c)
end

# Adapted from hops, include/hops/MarkovChain/Proposal/CoordinateHitAndRunProposal.hpp
# (commit b62592e, MIT; Copyright (c) 2020 Johann Fredrik Jadebeck, Axel Theorell, Samuel
# Leweke, Katharina Nöh): the slacks are updated after each move instead of recomputed,
# and the cyclic scan.
function step!(rng::AbstractRNG, c::HitAndRunState{CoordinateHitAndRun})
    w = c.warmup
    k = length(c.z)
    if k > 0
        refresh_slack!(c)
        cyclic = c.method.scan === :cyclic
        for _ in 1:something(c.method.moves, k)
            j = cyclic ? c.next_coordinate : rand(rng, 1:k)
            c.next_coordinate = j == k ? 1 : j + 1
            g = view(w.G, :, j)
            t = uniform(rng, chord(c.slack, g)...)
            c.z[j] += t
            for i in eachindex(c.slack)
                c.slack[i] -= t * g[i]
            end
        end
    end
    return finish_step!(c)
end

function update_slack!(c)
    mul!(c.slack, c.warmup.G, c.z)
    for i in eachindex(c.slack)
        c.slack[i] = c.warmup.h[i] - c.slack[i]
    end
    return c.slack
end

# Recompute the slacks on the first step and every `SLACK_REFRESH` steps after. The first
# step must recompute, as the chain may start from a state whose `z` was set by hand.
function refresh_slack!(c)
    c.steps % SLACK_REFRESH == 0 && update_slack!(c)
    c.steps += 1
    return c.slack
end

# Set `c.x = center + transform * c.z`
function update_x!(c)
    w = c.warmup
    mul!(c.x, w.transform, c.z)
    for i in eachindex(c.x)
        c.x[i] += w.center[i]
    end
    return c
end

function finish_step!(c::HitAndRunState)
    update_x!(c)
    c.n_seen += 1
    for i in eachindex(c.center)
        c.center[i] += (c.z[i] - c.center[i]) / c.n_seen
    end
    return c
end

# AbstractMCMC interface. The first step starts from the warmup center.
function AbstractMCMC.step(rng::AbstractRNG, w::Warmup, method::PolytopeSampler; kwargs...)
    c = step!(rng, chain_state(method, w))
    return copy(c.x), c
end

function AbstractMCMC.step(rng::AbstractRNG, ::Warmup, ::PolytopeSampler, c; kwargs...)
    step!(rng, c)
    return copy(c.x), c
end

# Return the points of a chain as the columns of a matrix, unless another `chain_type` is
# asked for
function AbstractMCMC.bundle_samples(
        samples::Vector{Vector{Float64}}, ::Warmup, ::PolytopeSampler, state, ::Type{Any};
        kwargs...,
    )
    X = Matrix{Float64}(undef, length(state.x), length(samples))
    for (j, x) in enumerate(samples)
        X[:, j] = x
    end
    return X
end

"""
    sample([rng], w::Warmup, method, n; discard_initial = 0, thinning = 1, progress = true)
    sample([rng], w::Warmup, method, MCMCThreads(), n, nchains; kwargs...)
    sample([rng], P::Polytope, method, args...; extremes = false, rounding = :ellipsoid,
           optimizer = HiGHS.Optimizer, kwargs...)

Draw `n` points from the uniform distribution on a polytope, returned as the columns of a
matrix; with `MCMCThreads()`, `MCMCSerial()` or `MCMCDistributed()`, draw `nchains` chains
and return a vector of matrices. `method` is [`CoordinateHitAndRun`](@ref),
[`BilliardWalk`](@ref), [`HitAndRun`](@ref) or [`ACHR`](@ref).

Every chain starts at the warmup center, discards `discard_initial` steps and then keeps
every `thinning`-th point. Consecutive points are correlated; check the effective sample
size with [`diagnostics`](@ref). Given a [`Polytope`](@ref), `sample` calls
[`warmup`](@ref) once for all chains with the given `extremes` and `rounding`. To run
chains until they converge, see [`sample_until_converged`](@ref).

This is `AbstractMCMC.sample` (also `StatsBase.sample`); its other keyword arguments, such
as `initial_state` and `callback`, apply.

# Examples
```julia
P = Polytope([0.0, 0.0], [1.0, 1.0]; A = [1.0 1.0], b = [1.0])  # a triangle
X = sample(Xoshiro(1), P, HitAndRun(), 1000; discard_initial = 100, thinning = 10)
chains = sample(Xoshiro(1), P, ACHR(), MCMCThreads(), 1000, 4; progress = false)
diagnostics(chains)
```
"""
function AbstractMCMC.sample(
        rng::AbstractRNG, P::Polytope, method::PolytopeSampler, n::Integer;
        extremes::Bool = false, rounding::Union{Bool, Symbol} = :ellipsoid,
        optimizer = HiGHS.Optimizer, kwargs...,
    )
    return sample(rng, warmup(rng, P; extremes, rounding, optimizer), method, n; kwargs...)
end

function AbstractMCMC.sample(
        rng::AbstractRNG, P::Polytope, method::PolytopeSampler,
        ensemble::AbstractMCMCEnsemble, n::Integer, nchains::Integer;
        extremes::Bool = false, rounding::Union{Bool, Symbol} = :ellipsoid,
        optimizer = HiGHS.Optimizer, kwargs...,
    )
    w = warmup(rng, P; extremes, rounding, optimizer)
    return sample(rng, w, method, ensemble, n, nchains; kwargs...)
end
