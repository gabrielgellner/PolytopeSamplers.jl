"""
    diagnostics(chains; kind = :bulk)

Effective sample size and split-R̂ of every coordinate, from one chain (a matrix with one
point per column) or several chains of equal length (a vector of such matrices), as
returned by [`sample`](@ref).

Returns a named tuple:

- `ess`, `rhat`: one value per coordinate, `NaN` for coordinates that are constant over
  every chain, such as fixed coordinates.
- `min_ess`, `max_rhat`: the smallest ESS and the largest R̂ over the varying coordinates.

`kind = :bulk` measures the center of each marginal and `kind = :tail` its 5% and 95%
quantiles (Vehtari et al. 2021). A common rule is to run until `max_rhat < 1.01` and
`min_ess` is at least 100 per chain. Computed with MCMCDiagnosticTools.jl.

Vehtari, A., Gelman, A., Simpson, D., Carpenter, B. & Bürkner, P.-C. (2021).
Rank-normalization, folding, and localization: an improved R̂ for assessing convergence of
MCMC. Bayesian Analysis 16, 667–718. <https://doi.org/10.1214/20-BA1221>
"""
function diagnostics(chains::AbstractVector{<:AbstractMatrix{<:Real}}; kind::Symbol = :bulk)
    isempty(chains) && throw(ArgumentError("no chains given"))
    allequal(size, chains) || throw(DimensionMismatch("chains must all have the same size"))
    nvars, ndraws = size(first(chains))
    ndraws >= 4 || throw(ArgumentError("chains need at least 4 points, got $ndraws"))
    # MCMCDiagnosticTools takes draws × chains × parameters
    x = Array{Float64}(undef, ndraws, length(chains), nvars)
    for (j, X) in enumerate(chains)
        x[:, j, :] = X'
    end
    (; ess, rhat) = ess_rhat(x; kind)
    varying = .!isnan.(ess)
    min_ess = any(varying) ? minimum(ess[varying]) : NaN
    max_rhat = any(varying) ? maximum(rhat[varying]) : NaN
    return (; ess, rhat, min_ess, max_rhat)
end

diagnostics(X::AbstractMatrix{<:Real}; kwargs...) = diagnostics([X]; kwargs...)
