# Many replicates of the constrained food web study in parallel, one per thread: a web,
# biomasses, growth rates redrawn until the equilibrium is feasible, chains until 1000
# effective samples, and the fraction of stable community matrices. Each replicate seeds
# its own generator, so the results do not depend on the number of threads.
# Run with `just example foodweb_replicates` (all threads).

include("foodweb_polytope.jl")

using LinearAlgebra: BLAS
using PolytopeSamplers: sample_until_converged, warmup

# Each replicate runs on one thread; BLAS threads would compete with them
BLAS.set_num_threads(1)

function replicate(seed; S = 15, C = 0.15)
    rng = Xoshiro(seed)
    web = generalized_cascade_network(rng, S, C; basal_paths = true)
    x = pyramid_biomass(rng, web)
    basal = basal_species(web)
    signs = interaction_signs(web)
    while true
        r = [i in basal ? 5 * rand(rng) : -rand(rng) for i in 1:S]
        P, links = foodweb_polytope(web, x, r)
        isempty(P) && continue
        w = warmup(rng, P; extremes = false)
        result = sample_until_converged(rng, w; target_ess = 1_000)
        λ = [spectral_abscissa(community_matrix(α, links, signs, x)) for X in result.chains for α in eachcol(X)]
        return mean(<(0), λ)
    end
    return
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
