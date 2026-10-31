# Interaction strengths of a food web consistent with a given equilibrium, after the
# constrained food webs of Gellner, McCann & Hastings
# (https://github.com/gabrielgellner/constrained-foodwebs).
#
# Lotka–Volterra dynamics `dx_i/dt = x_i (r_i + Σ_j a_ij x_j)` have an equilibrium at the
# biomasses `x` when `r_i + Σ_j a_ij x_j = 0` for every species. Given the web, `x` and `r`,
# the magnitudes `α_ij = |a_ij|` of the links that exist form a polytope:
#
# - one equality per species, `Σ_j s_ij x_j α_ij + r_i = 0`, with `s_ij` the sign of the
#   interaction (−1 for predation on `i` and for self-limitation, +1 for consumption by `i`);
# - box bounds `lb ≤ α_ij ≤ ub`;
# - with `energetic = true`, for each predator `j` of prey `i`, `α_ij ≥ α_ji`: a predator gains
#   less from a unit of prey than the prey loses.
#
# Uniform samples from this polytope are the community matrices consistent with the
# equilibrium. This file defines the polytope; `constrained_foodweb.jl` samples one web and
# `foodweb_replicates.jl` runs many in parallel.

using FoodWebs: basal_species, generalized_cascade_network, spectral_abscissa, top_species,
    trophic_levels
using PolytopeSamplers: Polytope
using Random: Xoshiro
using Statistics: mean

"""
    interaction_signs(web)

Signs `s[i, j]` of the effect of species `j` on species `i`: −1 on prey, +1 on predators,
and −1 on the diagonal except for top species, which have density-independent mortality.
`web[i, j] == 1` means `j` eats `i`.
"""
function interaction_signs(web)
    s = Int.(web') .- Int.(web)
    for i in axes(s, 1)
        s[i, i] = -1
    end
    for i in top_species(web)
        s[i, i] = 0
    end
    return s
end

"""
    foodweb_polytope(web, x, r; lb = 0.0, ub = 5.0, energetic = true)

The polytope of interaction magnitudes `α` with an equilibrium at biomasses `x` for growth
rates `r`, and the `(i, j)` link of each coordinate of `α`.
"""
function foodweb_polytope(web, x, r; lb = 0.0, ub = 5.0, energetic = true)
    s = interaction_signs(web)
    S = size(web, 1)
    links = [(i, j) for i in 1:S for j in 1:S if s[i, j] != 0]
    index = Dict(l => k for (k, l) in enumerate(links))
    n = length(links)

    C = zeros(S, n)
    for (k, (i, j)) in enumerate(links)
        C[i, k] = s[i, j] * x[j]
    end
    pairs = [(i, j) for i in 1:S for j in 1:S if web[i, j]]
    A = zeros(energetic ? length(pairs) : 0, n)
    if energetic
        for (row, (prey, pred)) in enumerate(pairs)
            A[row, index[(pred, prey)]] = 1   # α_pred,prey - α_prey,pred ≤ 0
            A[row, index[(prey, pred)]] = -1
        end
    end
    P = Polytope(fill(lb, n), fill(ub, n); A, b = zeros(size(A, 1)), C, d = -r)
    return P, links
end

# The Jacobian at equilibrium, J[i, j] = x_i a_ij
function community_matrix(α, links, signs, x)
    J = zeros(length(x), length(x))
    for (k, (i, j)) in enumerate(links)
        J[i, j] = x[i] * signs[i, j] * α[k]
    end
    return J
end

# Biomass falling by `δ` per trophic level from `top`, fuzzed by up to 20% (an Eltonian
# pyramid for δ > 0)
function pyramid_biomass(rng, web; top = 10.0, δ = 2.0)
    tl = trophic_levels(web) .- 1
    return top .+ δ .* (maximum(tl) .- tl) .* (0.8 .+ 0.2 .* rand(rng, length(tl)))
end
