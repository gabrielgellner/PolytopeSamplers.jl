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

out = []
@progress for S in 10:30
    #TODO: this preallocating to avoid scope issues in the while feels bad
    web = init_model(S, 0.41)
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

    chain = hr_sample!(sampler, 100_000; burn_in = 0)

    push!(out, [S, [λ_stability(cmat(web, c)) for c in sample(chain, 1000)]])
end

for o in out
    plot(fill(o[1], length(o[2])), o[2], "ko")
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

        chain = hr_sample!(sampler, 500_000; burn_in = 0)

        push!(out, mean([maximum(real.(eigvals(cmat(web, c)))) for c in chain[(end - 10000):end]]))
    end

    return out
end

@time out = same_S(30, nsamp = 60)

plot(cumsum(out) ./ collect(1:length(out)))
plot(out)


for i in 1:length(out)
    out[i]
end

for (i, o) in enumerate(out)
    plot(fill(i, length(o[2])), o[2] , "ko")
end
