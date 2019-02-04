using LinearAlgebra
using FoodWebs
using JLD2

include("sampler.jl")
include("biomass_tools.jl")

outputpath(name) = normpath(joinpath("C:/Users/Gabriel", "SimOutput/$name"))

function init_model(S, C; lb = 0.0, ub = 10.0, min_β = 10.0, δ = 0.0)
    pnet = cascade_network(S, C)
    β = set_β(pnet, δ, min_β)

    growth = fill(1.0, S)
    for s in basal_species(pnet)
        growth[s] = -1.0
    end
    return ConstrainedFoodWeb(pnet, β, growth, lb, ub)
end

function resample_stab(S, C; δ = 0.0, min_β = 10.0, nsamp = 200)
    webs = []
    chains = []
    @progress for i in 1:nsamp
        web = init_model(S, C, δ = δ, min_β = min_β)
        sampler = ACHRSampler(web)

        max_iter = 0
        while true
            web = init_model(S, C, δ = δ, min_β = min_β)
            sampler = ACHRSampler(web)

            status = warmup_sampler!(sampler)
            if status == :Feasible
                break
            else
                max_iter += 1
            end

            if max_iter > 100
                error("could not find feasible start")
            end
        end

        chain = hr_sample!(sampler, 10_000; burn_in = 0)
        push!(webs, web)
        push!(chains, chain)
    end

    # Save samples to file
    jldopen(outputpath("sims_$S.jld2"), "w") do file
        file["S"] = S
        file["slope"] = δ
        file["webs"] = webs
        file["αs"] = chains
    end

    return
end

for S in 18:30
    resample_stab(S, 0.25, δ = -1.0)
end

resample_stab(10, 0.25, δ = -1.0)
