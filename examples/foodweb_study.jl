# The settings of the constrained food webs of Gellner, McCann & Hastings
# (https://github.com/gabrielgellner/constrained-foodwebs), shared by `constrained_foodweb.jl`
# and `foodweb_replicates.jl`. FoodWebs.jl builds the polytope; code here moves there once it
# is general enough.

using FoodWebs: trophic_levels

# Every species but the top predators is self-regulated, a predator gains at most what its
# prey loses, and every interaction strength is in [0, 5]
const constraints = (; self_regulation = :all_but_top, efficiency = (0.0, 1.0), bounds = (0.0, 5.0))

# Biomass falling by `δ` per trophic level from `top`, fuzzed by up to 20% (an Eltonian
# pyramid for δ > 0), as in the original study
function pyramid_biomass(rng, web; top = 10.0, δ = 2.0)
    tl = trophic_levels(web) .- 1
    return top .+ δ .* (maximum(tl) .- tl) .* (0.8 .+ 0.2 .* rand(rng, length(tl)))
end
