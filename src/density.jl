"""
    Target(w::Warmup, ℓ; gradient = nothing)
    Target(P::Polytope, ℓ; gradient = nothing, kwargs...)

The distribution on a polytope with density proportional to `exp(ℓ(x))`, a model for
[`sample`](@ref) and [`sample_until_converged`](@ref). Given a [`Polytope`](@ref), the
other keyword arguments go to [`warmup`](@ref).

`ℓ` is a function of the vector `x` of the polytope's variables, or an object implementing
the LogDensityProblems.jl interface (`logdensity`, `dimension`, `capabilities`, and
`logdensity_and_gradient` for gradients). For a function, pass `gradient = (g, x) -> ...`,
which fills `g` with `∇ℓ(x)`, to use [`CRHMC`](@ref). `ℓ` is called from every chain, on
separate threads when Julia has them.

[`HitAndRun`](@ref) and [`CoordinateHitAndRun`](@ref) draw each point on the chord from
the density along it by slice sampling, so they leave the target invariant for any `ℓ`;
[`DikinWalk`](@ref) and [`VaidyaWalk`](@ref) add `ℓ` to their Metropolis ratio;
[`CRHMC`](@ref) adds `-ℓ` to its potential and needs the gradient. [`BilliardWalk`](@ref)
and [`ACHR`](@ref) sample only the uniform distribution.

# Examples
```julia
P = Polytope(zeros(3), ones(3); C = ones(1, 3), d = [1.0])  # the simplex
# Dirichlet(2, 3, 4) on the simplex
target = Target(P, x -> sum((a - 1) * log(xi) for (a, xi) in zip((2, 3, 4), x));
                gradient = (g, x) -> (g .= ((2, 3, 4) .- 1) ./ x))
r = sample_until_converged(Xoshiro(1), target, CRHMC())
```
"""
struct Target{L} <: AbstractModel
    warmup::Warmup
    logdensity::L
    Target{L}(warmup::Warmup, logdensity::L) where {L} = new{L}(warmup, logdensity)
end

function Target(w::Warmup, ℓ; gradient = nothing)
    n = nvars(w.polytope)
    ℓ = ℓ isa Function ? FunctionLogDensity(ℓ, gradient, n) : ℓ
    gradient === nothing || ℓ isa FunctionLogDensity ||
        throw(ArgumentError("gradient applies only when ℓ is a function"))
    LogDensityProblems.capabilities(ℓ) === nothing &&
        throw(ArgumentError("ℓ must be a function or implement the LogDensityProblems interface"))
    LogDensityProblems.dimension(ℓ) == n ||
        throw(DimensionMismatch("ℓ has dimension $(LogDensityProblems.dimension(ℓ)), the polytope has $n variables"))
    return Target{typeof(ℓ)}(w, ℓ)
end

function Target(P::Polytope, ℓ; gradient = nothing, kwargs...)
    return Target(warmup(P; kwargs...), ℓ; gradient)
end

function Base.show(io::IO, t::Target)
    print(io, "Target on a ", size(t.warmup.transform, 2), "-dimensional slice of a ")
    print(io, nvars(t.warmup.polytope), "-variable polytope")
    return
end

# A log density given as a function, with an optional in-place gradient
struct FunctionLogDensity{F, G}
    f::F
    gradient!::G
    n::Int
end

LogDensityProblems.dimension(d::FunctionLogDensity) = d.n
LogDensityProblems.capabilities(::Type{<:FunctionLogDensity}) = LogDensityOrder{1}()
function LogDensityProblems.capabilities(::Type{<:FunctionLogDensity{F, Nothing}}) where {F}
    return LogDensityOrder{0}()
end
LogDensityProblems.logdensity(d::FunctionLogDensity, x) = d.f(x)
function LogDensityProblems.logdensity_and_gradient(d::FunctionLogDensity, x)
    g = similar(x, Float64)
    d.gradient!(g, x)
    return d.f(x), g
end

# The geometry and the per-chain density of a model
geometry(w::Warmup) = w
geometry(t::Target) = t.warmup
chain_density(::Warmup) = nothing
chain_density(t::Target) = ZDensity(t)

# A log density in the coordinates `z` of a warmup, with its own buffer for `x`, so each
# chain has one
struct ZDensity{L}
    ℓ::L
    warmup::Warmup
    x::Vector{Float64}
end

ZDensity(t::Target) = ZDensity(t.logdensity, t.warmup, copy(t.warmup.center))

has_gradient(d::ZDensity) = LogDensityProblems.capabilities(d.ℓ) != LogDensityOrder{0}()

function to_x!(d::ZDensity, z)
    mul!(d.x, d.warmup.transform, z)
    d.x .+= d.warmup.center
    return d.x
end

logdensity_z(d::ZDensity, z) = Float64(LogDensityProblems.logdensity(d.ℓ, to_x!(d, z)))

# Add `scale * ∇_z ℓ(z) = scale * Tᵀ ∇_x ℓ(x)` to `out`; return ℓ(z)
function add_gradient_z!(out, d::ZDensity, z, scale)
    value, g = LogDensityProblems.logdensity_and_gradient(d.ℓ, to_x!(d, z))
    mul!(out, d.warmup.transform', g, scale, 1.0)
    return Float64(value)
end

# The longest run of shrinkage steps before a slice update gives up and stays put; each
# step at least halves the expected interval, so this is reached only for degenerate
# densities
const SLICE_MAX = 200

"""
    slice_on_chord(rng, logp, logp0, tmin, tmax)

Draw `t` from the density proportional to `exp(logp(t))` on `[tmin, tmax]`, a chord through
the current point `t = 0` whose log density is `logp0`, by slice sampling with shrinkage
(Neal 2003, Section 4.2.2, with the whole chord as the initial interval). Returns `t` and
`logp(t)`.

Neal, R. M. (2003). Slice sampling. Annals of Statistics 31, 705–767.
<https://doi.org/10.1214/aos/1056562461>
"""
function slice_on_chord(rng::AbstractRNG, logp, logp0, tmin, tmax)
    logy = logp0 - randexp(rng)
    for _ in 1:SLICE_MAX
        t = tmin + (tmax - tmin) * rand(rng)
        lp = logp(t)
        lp > logy && return t, lp
        if t < 0
            tmin = t
        else
            tmax = t
        end
    end
    return 0.0, logp0
end

# The log density at `z`, zero for the uniform distribution
logdensity_or_zero(::Nothing, z) = 0.0
logdensity_or_zero(d::ZDensity, z) = logdensity_z(d, z)
