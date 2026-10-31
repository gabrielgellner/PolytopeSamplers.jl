"""
    AbstractHitAndRun

Supertype of the hit-and-run methods, which differ in how they choose a direction:
[`HitAndRun`](@ref) and [`ACHR`](@ref).
"""
abstract type AbstractHitAndRun end

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
mutable struct Chain{M <: AbstractHitAndRun}
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

function Chain(method::AbstractHitAndRun, w::Warmup)
    n, k = size(w.basis)
    m = length(w.polytope.b)
    return Chain(
        method, w, copy(w.center), zeros(n), zeros(k), zeros(m), zeros(m), copy(w.center),
        size(w.points, 2),
    )
end

# Set `c.u` to a unit direction; return false when there is none, as for a single point
function direction!(rng::AbstractRNG, c::Chain{HitAndRun})
    isempty(c.z) && return false
    randn!(rng, c.z)
    mul!(c.u, c.warmup.basis, c.z)
    normalize!(c.u)
    return true
end

function direction!(rng::AbstractRNG, c::Chain{ACHR})
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
function chord(c::Chain)
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
function step!(rng::AbstractRNG, c::Chain)
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

"""
    hr_sample([rng], P::Polytope, n; optimizer = HiGHS.Optimizer, kwargs...)
    hr_sample([rng], w::Warmup, n; method = HitAndRun(), burn_in = n, thin = 1)

Draw `n` points from the uniform distribution on a polytope by hit-and-run, returned as the
columns of a matrix. Pass a [`Warmup`](@ref) from [`warmup`](@ref) to reuse its linear
programs across chains; given a [`Polytope`](@ref), `hr_sample` calls `warmup` first.

The chain starts at the warmup center, discards `burn_in` steps and then keeps every
`thin`-th point, for `burn_in + n * thin` steps in total. Consecutive points are correlated;
raise `thin` to reduce the correlation. `method` is [`HitAndRun`](@ref) or [`ACHR`](@ref).

# Examples
```julia
P = Polytope([0.0, 0.0], [1.0, 1.0]; A = [1.0 1.0], b = [1.0])  # a triangle
X = hr_sample(Xoshiro(1), P, 1000; thin = 10)
```
"""
function hr_sample(
        rng::AbstractRNG, w::Warmup, n::Integer;
        method::AbstractHitAndRun = HitAndRun(), burn_in::Integer = n, thin::Integer = 1,
    )
    n >= 0 || throw(ArgumentError("n must be non-negative, got $n"))
    burn_in >= 0 || throw(ArgumentError("burn_in must be non-negative, got $burn_in"))
    thin >= 1 || throw(ArgumentError("thin must be positive, got $thin"))
    c = Chain(method, w)
    for _ in 1:burn_in
        step!(rng, c)
    end
    X = Matrix{Float64}(undef, length(c.x), n)
    for j in 1:n
        for _ in 1:thin
            step!(rng, c)
        end
        X[:, j] = c.x
    end
    return X
end

function hr_sample(rng::AbstractRNG, P::Polytope, n::Integer; optimizer = HiGHS.Optimizer, kwargs...)
    return hr_sample(rng, warmup(P; optimizer), n; kwargs...)
end

hr_sample(P::Union{Polytope, Warmup}, n::Integer; kwargs...) = hr_sample(default_rng(), P, n; kwargs...)
