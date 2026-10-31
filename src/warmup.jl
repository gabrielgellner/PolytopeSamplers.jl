"""
    Warmup

The starting state of a hit-and-run chain on a polytope, built by [`warmup`](@ref).

- `polytope`: the [`Polytope`](@ref) sampled.
- `points`: the distinct extreme points of each coordinate, one per column.
- `center`: the mean of `points`, where chains start.
- `basis`: orthonormal columns spanning the directions in which the polytope has width.
"""
struct Warmup
    polytope::Polytope
    points::Matrix{Float64}
    center::Vector{Float64}
    basis::Matrix{Float64}
end

function Base.show(io::IO, w::Warmup)
    print(io, "Warmup with ", size(w.points, 2), " points on a ", size(w.basis, 2))
    print(io, "-dimensional slice of a ", nvars(w.polytope), "-variable polytope")
    return
end

"""
    warmup(P::Polytope; optimizer = HiGHS.Optimizer, atol = 1e-6)

Solve `2n` linear programs, minimizing and maximizing each of the `n` coordinates over `P`,
and return a [`Warmup`](@ref) for [`hr_sample`](@ref).

The extreme points also locate the constraints that hold with equality over all of `P`: a
coordinate whose range is within `atol` of zero, or an inequality whose slack is within
`atol` of zero at every extreme point. Chains move only in the directions that keep these
equalities and the equalities `C * x .== d`, as any other direction leaves `P` at once.

Throws an `ArgumentError` when `P` is empty or unbounded. `optimizer` is any JuMP optimizer
for linear programs; each call builds its own model, so separate tasks may call `warmup`
at the same time.
"""
function warmup(P::Polytope; optimizer = HiGHS.Optimizer, atol::Real = 1.0e-6)
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
    basis = direction_basis(Ceq, points, center, atol)
    return Warmup(P, points, center, basis)
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
