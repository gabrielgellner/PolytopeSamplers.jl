"""
    Polytope(lower, upper; A, b, C, d)

The convex polytope of points `x` with `lower .<= x .<= upper`, `A * x .<= b` and
`C * x .== d`. Bounds may be infinite as long as the constraints bound the polytope. A
coordinate with `lower[i] == upper[i]` is fixed at that value.

`A`, `b`, `C` and `d` default to no constraints. Constraints of the form `L * x .>= f` are
`A = -L, b = -f`.

`isempty(P)` solves a linear program to check whether `P` has any points.
"""
struct Polytope
    lower::Vector{Float64}
    upper::Vector{Float64}
    A::Matrix{Float64}
    b::Vector{Float64}
    C::Matrix{Float64}
    d::Vector{Float64}

    function Polytope(lower, upper, A, b, C, d)
        n = length(lower)
        length(upper) == n ||
            throw(DimensionMismatch("lower has length $n but upper has length $(length(upper))"))
        all(lower .<= upper) || throw(ArgumentError("lower bounds must not exceed upper bounds"))
        size(A) == (length(b), n) ||
            throw(DimensionMismatch("A must be $(length(b))×$n to match b and the bounds, got $(size(A))"))
        size(C) == (length(d), n) ||
            throw(DimensionMismatch("C must be $(length(d))×$n to match d and the bounds, got $(size(C))"))
        return new(lower, upper, A, b, C, d)
    end
end

function Polytope(
        lower::AbstractVector{<:Real}, upper::AbstractVector{<:Real};
        A::AbstractMatrix{<:Real} = zeros(0, length(lower)), b::AbstractVector{<:Real} = zeros(0),
        C::AbstractMatrix{<:Real} = zeros(0, length(lower)), d::AbstractVector{<:Real} = zeros(0),
    )
    return Polytope(lower, upper, A, b, C, d)
end

nvars(P::Polytope) = length(P.lower)

function Base.show(io::IO, P::Polytope)
    print(io, "Polytope in ", nvars(P), " variables with ", length(P.b), " inequalities and ")
    print(io, length(P.d), " equalities")
    return
end

"""
    isfeasible(P::Polytope, x; atol = 1e-8)

Whether `x` satisfies every constraint of `P` to within `atol`. `x in P` uses the default
tolerance.
"""
function isfeasible(P::Polytope, x::AbstractVector{<:Real}; atol::Real = 1.0e-8)
    length(x) == nvars(P) || throw(DimensionMismatch("x has length $(length(x)), P has $(nvars(P)) variables"))
    all(P.lower .- atol .<= x .<= P.upper .+ atol) || return false
    all(P.A * x .<= P.b .+ atol) || return false
    return all(abs.(P.C * x .- P.d) .<= atol)
end

Base.in(x::AbstractVector{<:Real}, P::Polytope) = isfeasible(P, x)

# A JuMP model with the constraints of `P` and no objective. Constraints are built from the
# nonzero coefficients, as the matrices of food web problems are mostly zeros.
function lp_model(P::Polytope, optimizer)
    model = Model(optimizer)
    set_silent(model)
    n = nvars(P)
    x = @variable(model, [1:n])
    for i in 1:n
        isfinite(P.lower[i]) && set_lower_bound(x[i], P.lower[i])
        isfinite(P.upper[i]) && set_upper_bound(x[i], P.upper[i])
    end
    for i in eachindex(P.b)
        @constraint(model, sum(P.A[i, j] * x[j] for j in 1:n if P.A[i, j] != 0; init = 0.0) <= P.b[i])
    end
    for i in eachindex(P.d)
        @constraint(model, sum(P.C[i, j] * x[j] for j in 1:n if P.C[i, j] != 0; init = 0.0) == P.d[i])
    end
    return model, x
end

"""
    isempty(P::Polytope; optimizer = HiGHS.Optimizer)

Whether `P` has no points, decided by a linear program.
"""
function Base.isempty(P::Polytope; optimizer = HiGHS.Optimizer)
    model, _ = lp_model(P, optimizer)
    optimize!(model)
    is_solved_and_feasible(model) && return false
    # With no objective the problem cannot be unbounded, so either status means infeasible
    status = termination_status(model)
    status in (INFEASIBLE, INFEASIBLE_OR_UNBOUNDED) && return true
    return error("the feasibility linear program stopped with status $status")
end
