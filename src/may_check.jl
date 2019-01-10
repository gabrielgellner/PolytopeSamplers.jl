using LinearAlgebra
using FoodWebs
using MCMCDiagnostics
using PyPlot

include("sampler.jl")

function init_model(S, C; lb = 0.0, ub = 10.0, biomass = nothing)
    pnet = cascade_network(S, C)
    if biomass === nothing
        biomass = rand(S)
    end
    growth = fill(1.0, S)
    for s in basal_species(pnet)
        growth[s] = -1.0
    end
    return ConstrainedFoodWeb(pnet, copy(biomass), growth, lb, ub)
end

function same_S(S; nsamp = 10)
    out = []
    @progress for i in 1:nsamp
        biomass = rand(S)
        #TODO: this preallocating to avoid scope issues in the while feels bad
        web = init_model(S, 0.41, biomass = biomass)
        sampler = ACHRSampler(web)

        max_iter = 0
        while true
            web = init_model(S, 0.41)
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

        chain = hr_sample!(sampler, 1_000_000; burn_in = 0)

        push!(out, mean([maximum(real.(eigvals(cmat(web, c)))) for c in chain[(end - 100000):end]]))
    end

    return out
end

function div_stab()
    out = []
    @progress for S in 10:30
        push!(out, [S, mean(same_S(S, nsamp = 100))])
    end

    return VectorOfArray(out)
end

@time out = div_stab()

plot(out[1, :], out[2, :], "ko")
