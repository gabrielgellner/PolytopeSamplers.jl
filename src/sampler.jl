"""
    AbstractHitAndRun

Supertype of the hit-and-run methods, which differ in how they choose a direction:
[`HitAndRun`](@ref) and [`ACHR`](@ref).
"""
abstract type AbstractHitAndRun <: AbstractSampler end

"""
    HitAndRun()

Hit-and-run with directions drawn uniformly from the unit sphere in the span of the
polytope (Smith 1984). The chain is Markov and its stationary distribution is uniform on
the polytope.
"""
struct HitAndRun <: AbstractHitAndRun end

"""
    ACHR()

Artificial centering hit-and-run (Kaufman & Smith 1998), as in the COBRA Toolbox: each
direction runs from the running mean of the chain to a random warmup point. Directions
follow the shape of the polytope, so long thin polytopes mix faster than with
[`HitAndRun`](@ref). The direction depends on the history through the mean, so the chain
is only approximately Markov; its distribution approaches uniform as the mean settles.

Kaufman, D. E. & Smith, R. L. (1998). Direction choice for accelerated convergence in
hit-and-run sampling. Operations Research 46, 84–95. <https://doi.org/10.1287/opre.46.1.84>
"""
struct ACHR <: AbstractHitAndRun end

# Coefficients below this are treated as zero: a direction component this small moves a
# fixed coordinate or an equality-tight inequality by rounding error only.
const ZERO_TOL = 1.0e-12

# The state of one chain, with buffers so that a step does not allocate
mutable struct HitAndRunState{M <: AbstractHitAndRun}
    const method::M
    const warmup::Warmup
    const x::Vector{Float64}
    const u::Vector{Float64} # direction
    const z::Vector{Float64} # direction in basis coordinates
    const Ax::Vector{Float64}
    const Au::Vector{Float64}
    const center::Vector{Float64} # running mean, for ACHR
    n_seen::Int # points in the running mean
end

function HitAndRunState(method::AbstractHitAndRun, w::Warmup)
    n, k = size(w.basis)
    m = length(w.polytope.b)
    return HitAndRunState(
        method, w, copy(w.center), zeros(n), zeros(k), zeros(m), zeros(m), copy(w.center),
        size(w.points, 2),
    )
end

# Set `c.u` to a unit direction; return false when there is none, as for a single point
function direction!(rng::AbstractRNG, c::HitAndRunState{HitAndRun})
    isempty(c.z) && return false
    randn!(rng, c.z)
    mul!(c.u, c.warmup.basis, c.z)
    normalize!(c.u)
    return true
end

function direction!(rng::AbstractRNG, c::HitAndRunState{ACHR})
    B = c.warmup.basis
    isempty(c.z) && return false
    j = rand(rng, axes(c.warmup.points, 2))
    for i in eachindex(c.u)
        c.u[i] = c.warmup.points[i, j] - c.center[i]
    end
    # Project onto the basis so that rounding in the running mean does not accumulate
    # in directions that leave the polytope
    mul!(c.z, B', c.u)
    mul!(c.u, B, c.z)
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

# The interval of `t` with `x + t * u` in the polytope. Slacks are clamped at zero so that a
# point that rounding has put just outside a constraint still gets an interval holding zero.
function chord(c::HitAndRunState)
    P = c.warmup.polytope
    x, u = c.x, c.u
    tmin, tmax = -Inf, Inf
    for i in eachindex(x)
        tmin, tmax = narrow(tmin, tmax, u[i], max(P.upper[i] - x[i], 0.0))
        tmin, tmax = narrow(tmin, tmax, -u[i], max(x[i] - P.lower[i], 0.0))
    end
    mul!(c.Ax, P.A, x)
    mul!(c.Au, P.A, u)
    for i in eachindex(c.Ax)
        tmin, tmax = narrow(tmin, tmax, c.Au[i], max(P.b[i] - c.Ax[i], 0.0))
    end
    return tmin, tmax
end

# One hit-and-run step: move `c.x` to a uniform point on the chord through it along a new
# direction, and update the running mean
function step!(rng::AbstractRNG, c::HitAndRunState)
    if direction!(rng, c)
        tmin, tmax = chord(c)
        isfinite(tmin) && isfinite(tmax) ||
            throw(ArgumentError("the polytope is unbounded along a sampled direction"))
        t = tmin + (tmax - tmin) * rand(rng)
        for i in eachindex(c.x)
            c.x[i] += t * c.u[i]
        end
    end
    c.n_seen += 1
    for i in eachindex(c.center)
        c.center[i] += (c.x[i] - c.center[i]) / c.n_seen
    end
    return c
end

# AbstractMCMC interface. The first step starts from the warmup center.
function AbstractMCMC.step(rng::AbstractRNG, w::Warmup, method::AbstractHitAndRun; kwargs...)
    c = step!(rng, HitAndRunState(method, w))
    return copy(c.x), c
end

function AbstractMCMC.step(
        rng::AbstractRNG, ::Warmup, ::AbstractHitAndRun, c::HitAndRunState; kwargs...,
    )
    step!(rng, c)
    return copy(c.x), c
end

# Return the points of a chain as the columns of a matrix, unless another `chain_type` is
# asked for
function AbstractMCMC.bundle_samples(
        samples::Vector{Vector{Float64}}, ::Warmup, ::AbstractHitAndRun, state, ::Type{Any};
        kwargs...,
    )
    return reduce(hcat, samples; init = zeros(length(state.x), 0))
end

"""
    sample([rng], w::Warmup, method, n; discard_initial = 0, thinning = 1, progress = true)
    sample([rng], w::Warmup, method, MCMCThreads(), n, nchains; kwargs...)
    sample([rng], P::Polytope, method, args...; optimizer = HiGHS.Optimizer, kwargs...)

Draw `n` points from the uniform distribution on a polytope, returned as the columns of a
matrix; with `MCMCThreads()`, `MCMCSerial()` or `MCMCDistributed()`, draw `nchains` chains
and return a vector of matrices. `method` is [`HitAndRun`](@ref) or [`ACHR`](@ref).

Every chain starts at the warmup center, discards `discard_initial` steps and then keeps
every `thinning`-th point. Consecutive points are correlated; check the effective sample
size with [`diagnostics`](@ref). Given a [`Polytope`](@ref), `sample` calls
[`warmup`](@ref) once for all chains.

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
        rng::AbstractRNG, P::Polytope, method::AbstractHitAndRun, n::Integer;
        optimizer = HiGHS.Optimizer, kwargs...,
    )
    return sample(rng, warmup(P; optimizer), method, n; kwargs...)
end

function AbstractMCMC.sample(
        rng::AbstractRNG, P::Polytope, method::AbstractHitAndRun,
        ensemble::AbstractMCMCEnsemble, n::Integer, nchains::Integer;
        optimizer = HiGHS.Optimizer, kwargs...,
    )
    return sample(rng, warmup(P; optimizer), method, ensemble, n, nchains; kwargs...)
end
