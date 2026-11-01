# Many replicates of the constrained food web study in parallel, one per thread: a web,
# steady-state biomasses, growth rates redrawn until the equilibrium is feasible, billiard-walk
# chains until 1000 effective samples, and the fraction of stable community matrices. Each
# replicate seeds its own generator, so the results do not depend on the number of threads.
# Run with `just example foodweb_replicates` (all threads).

using FoodWebs: feasible_growth_rates, generalized_cascade_network, sample_communities,
    spectral_abscissa, steady_state_biomass
using LinearAlgebra: BLAS
using PolytopeSamplers: PolytopeSamplers
using Random: Xoshiro
using Statistics: mean

# Each replicate runs on one thread; BLAS threads would compete with them
BLAS.set_num_threads(1)

# The constraints of the constrained food webs of Gellner, McCann & Hastings; see
# `constrained_foodweb.jl`
const constraints = (; self_regulation = :all_but_top, efficiency = (0.0, 1.0), bounds = (0.0, 5.0))

function replicate(seed; S = 15, C = 0.15)
    rng = Xoshiro(seed)
    web = generalized_cascade_network(rng, S, C; basal_paths = true)
    x = 20 .* steady_state_biomass(web; efficiency = 0.5)
    (; communities) = feasible_growth_rates(rng, web, x; constraints...)
    result = sample_communities(rng, communities; target_ess = 1_000)
    return mean(spectral_abscissa(c.J) < 0 for c in result.communities)
end

let N = 80
    replicate(1) # compile before timing
    fractions = zeros(N)
    t = @elapsed Threads.@threads for i in 1:N
        fractions[i] = replicate(100 + i)
    end
    println(
        N, " replicates on ", Threads.nthreads(), " threads in ", round(t; digits = 1),
        " s; mean stable fraction ", round(mean(fractions); digits = 4),
    )
end
