# Sample the interaction strengths of one food web with each sampler and compare them.
# FoodWebs.jl builds the polytope: `FeasibleCommunities` holds the Lotka–Volterra parameters
# with an equilibrium at the biomasses `x`, and its PolytopeSamplers.jl extension converts it
# to a `Polytope`. Run with `just example constrained_foodweb`.

using FoodWebs: community_matrix, feasible_growth_rates, generalized_cascade_network,
    spectral_abscissa, steady_state_biomass
using PolytopeSamplers: ACHR, BilliardWalk, CoordinateHitAndRun, HitAndRun, MCMCThreads,
    Polytope, diagnostics, mmcs, sample, warmup
using Random: Xoshiro
using Statistics: mean

# The constraints of the constrained food webs of Gellner, McCann & Hastings
# (https://github.com/gabrielgellner/constrained-foodwebs): every species but the top
# predators is self-regulated, a predator gains at most what its prey loses, and every
# interaction strength is in [0, 5]
const constraints = (; self_regulation = :all_but_top, efficiency = (0.0, 1.0), bounds = (0.0, 5.0))

let rng = Xoshiro(2026), S = 15
    web = generalized_cascade_network(rng, S, 0.15; basal_paths = true)
    x = 20 .* steady_state_biomass(web; efficiency = 0.5)

    # Growth rates redrawn until the equilibrium is feasible
    (; communities, draws) = feasible_growth_rates(rng, web, x; constraints...)
    P = Polytope(communities)
    stable_fraction(chains) =
        mean(spectral_abscissa(community_matrix(communities, θ)) < 0 for X in chains for θ in eachcol(X))

    unrounded = warmup(P; rounding = false)
    t_round = @elapsed rounded = warmup(P)
    println("feasible growth rates after $draws draws; ", rounded)
    println("rounding took ", round(t_round; digits = 2), " s")

    # Four chains per sampler; compare effective samples per second
    runs = [
        ("HitAndRun, unrounded", unrounded, HitAndRun()),
        ("ACHR, unrounded", unrounded, ACHR()),
        ("HitAndRun", rounded, HitAndRun()),
        ("ACHR", rounded, ACHR()),
        ("CoordinateHitAndRun", rounded, CoordinateHitAndRun()),
        ("BilliardWalk", rounded, BilliardWalk()),
    ]
    for (name, w, method) in runs
        t = @elapsed chains = sample(
            rng, w, method, MCMCThreads(), 2_000, 4; discard_initial = 1_000, thinning = 10,
            progress = false,
        )
        d = diagnostics(chains)
        println(
            rpad(name, 22), " min ESS ", lpad(round(Int, d.min_ess), 5), ", max R̂ ",
            rpad(round(d.max_rhat; digits = 3), 5), ", ", lpad(round(Int, d.min_ess / t), 6),
            " ESS/s, stable fraction ", round(stable_fraction(chains); digits = 3),
        )
    end

    # Multiphase Monte Carlo sampling: one LP, rounding from samples, chains until
    # 1000 effective samples with R̂ < 1.01
    t = @elapsed r = mmcs(rng, P; target_ess = 1_000)
    println(
        rpad("mmcs", 22), " min ESS ", lpad(round(Int, r.diagnostics.min_ess), 5), ", max R̂ ",
        rpad(round(r.diagnostics.max_rhat; digits = 3), 5), ", ", round(t; digits = 2),
        " s in all, stable fraction ", round(stable_fraction(r.chains); digits = 3),
    )
end
