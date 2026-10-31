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
end

function Base.show(io::IO, w::Warmup)
    print(io, "Warmup with ", size(w.points, 2), " points on a ", size(w.transform, 2))
    print(io, "-dimensional slice of a ", nvars(w.polytope), "-variable polytope")
    w.rounded && print(io, ", rounded")
    return
end

"""
    warmup(P::Polytope; rounding = true, optimizer = HiGHS.Optimizer, atol = 1e-6)

Solve `2n` linear programs, minimizing and maximizing each of the `n` coordinates over `P`,
and return a [`Warmup`](@ref) to [`sample`](@ref) from.

With `rounding = true`, change coordinates so that the polytope is close to round: the
maximum-volume inscribed ellipsoid is mapped to the unit ball, repeatedly, until it is
nearly a ball. Rounding costs a few dense factorizations of size the number of constraints,
and makes every sampler take long steps in thin directions.

The extreme points also locate the constraints that hold with equality over all of `P`: a
coordinate whose range is within `atol` of zero, or an inequality whose slack is within
`atol` of zero at every extreme point. Chains move only in the directions that keep these
equalities and the equalities `C * x .== d`, as any other direction leaves `P` at once.

Throws an `ArgumentError` when `P` is empty or unbounded. `optimizer` is any JuMP optimizer
for linear programs; each call builds its own model, so separate tasks may call `warmup`
at the same time.
"""
function warmup(P::Polytope; rounding::Bool = true, optimizer = HiGHS.Optimizer, atol::Real = 1.0e-6)
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
    points = unique(points; dims = 2)

    Ceq, deq = active_equalities(P, points, atol)
    center = vec(mean(points; dims = 2))
    # Move the center onto the equalities, which the linear programs satisfy only to the
    # solver's tolerance
    if !isempty(deq)
        center .-= pinv(Ceq) * (Ceq * center .- deq)
    end
    T = direction_basis(Ceq, points, center, atol)
    G, h = reduced_constraints(P, center, T)
    if rounding
        round_polytope!(center, T, G, h; optimizer)
    end
    zpoints = isempty(T) ? zeros(0, size(points, 2)) : T \ (points .- center)
    radius = isempty(T) ? 0.0 : last(chebyshev_center(G, h, optimizer))
    return Warmup(P, points, center, T, G, h, zpoints, radius, rounding)
end

# The bounds and inequalities of `P` as `G z ≤ h` for `x = center + T z`, with unit-norm rows.
# Rows that are zero on the slice (fixed coordinates, inequalities that hold with equality)
# are left out, and slacks are clamped at zero against rounding error in `center`.
function reduced_constraints(P::Polytope, center, T)
    k = size(T, 2)
    rows = Vector{Float64}[]
    rhs = Float64[]
    function add!(g, s, scale)
        r = norm(g)
        if r > ROW_TOL * scale
            push!(rows, g ./ r)
            push!(rhs, max(s, 0.0) / r)
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
