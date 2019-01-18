using LinearAlgebra
using FoodWebs
using MCMCDiagnostics
using PyPlot

include("sampler.jl")
include("biomass_tools.jl")

function init_model(S, C; lb = 0.0, ub = 10.0, min_β = 10.0, δ = 0.0)
    pnet = cascade_network(S, C)
    β = set_β(pnet, δ, min_β)

    growth = fill(1.0, S)
    for s in basal_species(pnet)
        growth[s] = -1.0
    end
    return ConstrainedFoodWeb(pnet, β, growth, lb, ub)
end

out = []
S = 20
C = 0.3
δ = 0.5
min_β = 10.0
@progress for i in 1:100
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

    chain = hr_sample!(sampler, 300_000; burn_in = 0)
    push!(out, mean([maximum(real.(eigvals(cmat(web, c)))) for c in chain[(end - 100000):end]]))
end

plot(1:length(out), cumsum(out) ./ collect(1:length(out)))
