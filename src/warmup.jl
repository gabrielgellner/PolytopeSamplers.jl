# Values computed on first use and shared by the chains of a warmup, under the lock: Gᵀ and
# G Gᵀ for the billiard walk, and the extreme points in `z` for ACHR when the warmup was
# built without them
mutable struct WarmupCache
    reflections::Union{Nothing, Tuple{Matrix{Float64}, Matrix{Float64}}}
    zpoints::Union{Nothing, Matrix{Float64}}
    const optimizer::Any
    const lock::ReentrantLock
end

"""
    Warmup

A polytope in the coordinates that samplers move in, built by [`warmup`](@ref). Points are
`x = center + transform * z` for `z` in `{z : G z ≤ h}`, where `z` has one coordinate per
direction in which the polytope has width and the rows of `G` have unit norm.

- `polytope`: the [`Polytope`](@ref) sampled.
- `points`: the distinct extreme points of each coordinate, one per column.
- `center`: where chains start (`z = 0`); the center of the last inscribed ellipsoid when
  rounded, the mean of `points` otherwise.
- `transform`: the `n × k` matrix taking `z` to `x - center`. Its columns are orthonormal
  when not rounded.
- `G`, `h`: the constraints of `P` in `z`, leaving out those that are constant on the slice.
- `zpoints`: `points` in `z` coordinates.
- `radius`: the radius of the largest ball in `{z : G z ≤ h}`, near 1 when rounded.
- `rounded`: whether `transform` rounds the polytope.

With `warmup(P; extremes = false)` (the default), `points` and `zpoints` have no columns;
[`ACHR`](@ref) computes the extreme points when a chain first needs them.
"""
struct Warmup <: AbstractModel
    polytope::Polytope
    points::Matrix{Float64}
    center::Vector{Float64}
    transform::Matrix{Float64}
    G::Matrix{Float64}
    h::Vector{Float64}
    zpoints::Matrix{Float64}
    radius::Float64
    rounded::Bool
    cache::WarmupCache
end

function Warmup(
        P, points, center, transform, G, h, zpoints, radius, rounded;
        optimizer = HiGHS.Optimizer,
    )
    return Warmup(
        P, points, center, transform, G, h, zpoints, radius, rounded,
        WarmupCache(nothing, nothing, optimizer, ReentrantLock()),
    )
end

# Gᵀ and G Gᵀ, computed once per warmup, so that chains on separate threads share one copy
# instead of each holding an m × m matrix
function reflection_matrices(w::Warmup)
    cache = w.cache
    return lock(cache.lock) do
        if cache.reflections === nothing
            cache.reflections = (Matrix(w.G'), w.G * w.G')
        end
        cache.reflections
    end
end

# The extreme points of each coordinate in `z`: `w.zpoints`, or, for a warmup built without
# them, the `2n` linear programs solved once on first use. A warmup built from the largest
# inscribed ball has no equalities beyond those of its slice, so the points lie in it.
function extreme_zpoints(w::Warmup)
    isempty(w.zpoints) || return w.zpoints
    size(w.transform, 2) == 0 && return w.zpoints
    cache = w.cache
    return lock(cache.lock) do
        if cache.zpoints === nothing
            points = coordinate_extremes(w.polytope, cache.optimizer)
            cache.zpoints = w.transform \ (points .- w.center)
        end
        cache.zpoints
    end
end

function Base.show(io::IO, w::Warmup)
    print(io, "Warmup with ", size(w.points, 2), " points on a ", size(w.transform, 2))
    print(io, "-dimensional slice of a ", nvars(w.polytope), "-variable polytope")
    w.rounded && print(io, ", rounded")
    return
end

"""
    warmup([rng], P::Polytope; extremes = false, rounding = :ellipsoid,
           optimizer = HiGHS.Optimizer, atol = 1e-6)

Prepare `P` for sampling and return a [`Warmup`](@ref) to [`sample`](@ref) from.

With `extremes = false` (the default), solve one linear program for the largest ball inside
`P` in the slice of `C * x .== d` and the fixed coordinates, as in multiphase Monte Carlo
sampling ([`mmcs`](@ref)). A polytope with room for a ball of positive radius has no other
equalities. If the ball has radius below `atol`, `P` has equalities beyond the explicit
ones, and `warmup` falls back to `extremes = true`.

With `extremes = true`, solve `2n` linear programs, minimizing and maximizing each of the
`n` coordinates. The extreme points locate the constraints that hold with equality over all
of `P`: a coordinate whose range is within `atol` of zero, or an inequality whose slack is
within `atol` of zero at every extreme point. Chains move only in the directions that keep
these equalities and `C * x .== d`. The `2n` programs cost far more for large `n`: 58 s for
`n = 200` with 1000 constraints, against 1.3 s for the whole warmup with `extremes =
false`. [`ACHR`](@ref) uses the extreme points, and solves the `2n` programs on first use
when the warmup was built without them.

`rounding` changes coordinates so that the polytope is close to round, which makes every
sampler take long steps in thin directions:

- `:ellipsoid` (or `true`): map the maximum-volume inscribed ellipsoid to the unit ball,
  repeatedly, until it is nearly a ball. Each round factors dense matrices of size the
  number of constraints.
- `:samples`: draw billiard-walk samples and map their principal axes to equal lengths,
  repeatedly, as in multiphase Monte Carlo sampling. Uses `rng`.
- `:none` (or `false`).

Throws an `ArgumentError` when `P` is empty or unbounded; with `extremes = false` one more
linear program checks boundedness. `optimizer` is any JuMP
optimizer for linear programs; each call builds its own model, so separate tasks may call
`warmup` at the same time.
"""
function warmup(
        rng::AbstractRNG, P::Polytope; extremes::Bool = false,
        rounding::Union{Bool, Symbol} = :ellipsoid, optimizer = HiGHS.Optimizer,
        atol::Real = 1.0e-6,
    )
    rounding = rounding === true ? :ellipsoid : rounding === false ? :none : rounding
    rounding in (:ellipsoid, :samples, :none) ||
        throw(ArgumentError("rounding must be :ellipsoid, :samples or :none, got :$rounding"))
    n = nvars(P)
    if extremes
        points, center, T = extreme_slice(P, optimizer, atol)
        G, h = reduced_constraints(P, center, T)
    else
        slice = chebyshev_slice(P, optimizer, atol)
        slice === nothing && return warmup(rng, P; extremes = true, rounding, optimizer, atol)
        center, T, G, h = slice
        points = zeros(n, 0)
    end
    if rounding === :ellipsoid
        round_polytope!(center, T, G, h; optimizer)
    elseif rounding === :samples
        round_by_samples!(rng, P, center, T, G, h; optimizer)
    end
    zpoints = isempty(T) ? zeros(0, size(points, 2)) : T \ (points .- center)
    radius = isempty(T) ? 0.0 : last(chebyshev_center(G, h, optimizer))
    return Warmup(P, points, center, T, G, h, zpoints, radius, rounding !== :none; optimizer)
end

warmup(P::Polytope; kwargs...) = warmup(default_rng(), P; kwargs...)

# The distinct extreme points of each coordinate, one per column, from `2n` linear programs
function coordinate_extremes(P::Polytope, optimizer)
    model, x = lp_model(P, optimizer)
    optimize!(model)
    is_solved_and_feasible(model) || throw(ArgumentError("the polytope is empty"))

    n = nvars(P)
    points = Matrix{Float64}(undef, n, 2n)
    for i in 1:n, (k, sense) in enumerate((MIN_SENSE, MAX_SENSE))
        @objective(model, sense, x[i])
        optimize!(model)
        if !is_solved_and_feasible(model)
            throw(ArgumentError("the polytope is unbounded in coordinate $i (status $(termination_status(model)))"))
        end
        points[:, 2(i - 1) + k] = value.(x)
    end
    return unique(points; dims = 2)
end

# The extreme points of each coordinate, a center on the equalities, and a basis of the
# directions in which the polytope has width
function extreme_slice(P::Polytope, optimizer, atol)
    points = coordinate_extremes(P, optimizer)
    Ceq, deq = active_equalities(P, points, atol)
    center = vec(mean(points; dims = 2))
    # Move the center onto the equalities, which the linear programs satisfy only to the
    # solver's tolerance
    if !isempty(deq)
        center .-= pinv(Ceq) * (Ceq * center .- deq)
    end
    T = direction_basis(Ceq, points, center, atol)
    return points, center, T
end

# The slice of `C * x .== d` and the fixed coordinates, centered on the largest ball inside
# the polytope in that slice, from one linear program. Returns `nothing` when the ball has
# radius below `atol`, as the polytope then has further equalities.
function chebyshev_slice(P::Polytope, optimizer, atol)
    n = nvars(P)
    fixed = findall(P.lower .== P.upper)
    E = Matrix{Float64}(I, n, n)
    Ceq = [P.C; E[fixed, :]]
    deq = [P.d; P.lower[fixed]]
    x0 = isempty(deq) ? zeros(n) : pinv(Ceq) * deq
    if !isempty(deq) && norm(Ceq * x0 .- deq) > atol * max(1, norm(deq))
        throw(ArgumentError("the polytope is empty: its equalities have no solution"))
    end
    T = isempty(deq) ? E : nullspace(Ceq)
    T[abs.(T) .< ZERO_TOL] .= 0
    G, h = reduced_constraints(P, x0, T; clamp = false)
    if size(T, 2) == 0
        isfeasible(P, x0; atol) || throw(ArgumentError("the polytope is empty"))
        return x0, T, G, max.(h, 0.0)
    end
    c, r = chebyshev_center(G, h, optimizer)
    r > atol || return nothing
    isbounded(G, optimizer) || throw(ArgumentError("the polytope is unbounded"))
    return shift!(x0, T, G, h, c)
end

# Whether `{z : G z ≤ h}` (nonempty) is bounded: no direction `d ≠ 0` has `G d ≤ 0`. That
# holds exactly when `G` has full column rank, ruling out `G d = 0`, and some `y > 0` has
# `Gᵀ y = 0`, ruling out `G d ≤ 0` with `G d ≠ 0` (Stiemke's theorem of the alternative);
# the second is a linear feasibility problem in `y ≥ 1`
function isbounded(G, optimizer)
    m, k = size(G)
    m == 0 && return k == 0
    rank(G) == k || return false
    model = Model(optimizer)
    set_silent(model)
    y = @variable(model, [1:m], lower_bound = 1)
    @constraint(model, G' * y .== 0)
    optimize!(model)
    return is_solved_and_feasible(model)
end

# The bounds and inequalities of `P` as `G z ≤ h` for `x = center + T z`, with unit-norm rows.
# Rows that are zero on the slice (fixed coordinates, inequalities that hold with equality)
# are left out, and slacks are clamped at zero against rounding error in `center` unless
# `clamp = false` (when `center` need not be inside).
function reduced_constraints(P::Polytope, center, T; clamp::Bool = true)
    k = size(T, 2)
    rows = Vector{Float64}[]
    rhs = Float64[]
    function add!(g, s, scale)
        r = norm(g)
        if r > ROW_TOL * scale
            push!(rows, g ./ r)
            push!(rhs, (clamp ? max(s, 0.0) : s) / r)
        end
        return
    end
    for i in 1:nvars(P)
        g = T[i, :]
        isfinite(P.upper[i]) && add!(g, P.upper[i] - center[i], 1.0)
        isfinite(P.lower[i]) && add!(-g, center[i] - P.lower[i], 1.0)
    end
    for i in eachindex(P.b)
        a = P.A[i, :]
        add!(T' * a, P.b[i] - dot(a, center), norm(a))
    end
    G = isempty(rows) ? zeros(0, k) : permutedims(reduce(hcat, rows))
    return G, rhs
end

# The equalities `C * x .== d` with the fixed coordinates and the inequalities tight at every
# extreme point added as rows.
function active_equalities(P::Polytope, points, atol)
    n = nvars(P)
    rows = [P.C]
    rhs = [P.d]
    for i in 1:n
        lo, hi = extrema(view(points, i, :))
        if hi - lo <= atol
            e = zeros(1, n)
            e[i] = 1
            push!(rows, e)
            push!(rhs, [(lo + hi) / 2])
        end
    end
    slack = P.b .- P.A * points
    for i in eachindex(P.b)
        if maximum(view(slack, i, :)) <= atol
            push!(rows, P.A[i:i, :])
            push!(rhs, P.b[i:i])
        end
    end
    return reduce(vcat, rows), reduce(vcat, rhs)
end

# A constraint row whose norm on the slice is below this fraction of its norm in `x` is
# constant on the slice
const ROW_TOL = 1.0e-9

# Orthonormal directions that keep `Ceq * x` fixed and along which the extreme points spread.
# The second condition catches equalities that hold over the polytope but are not a single
# constraint row. Entries below `ZERO_TOL` are set to zero, so that a direction moves a fixed
# coordinate by exactly zero. A singular value counts as spread when it exceeds the
# `sqrt(m) * atol` that `m` points each within `atol` of a hyperplane can reach.
function direction_basis(Ceq, points, center, atol)
    n = length(center)
    Z = isempty(Ceq) ? Matrix{Float64}(I, n, n) : nullspace(Ceq)
    F = svd(Z' * (points .- center))
    k = count(>(sqrt(size(points, 2)) * atol), F.S)
    basis = Z * F.U[:, 1:k]
    basis[abs.(basis) .< ZERO_TOL] .= 0
    return basis
end
