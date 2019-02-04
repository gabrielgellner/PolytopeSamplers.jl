using LinearAlgebra
using StatsBase
using JLD2
using FileIO
using FoodWebs
using PyPlot

include("sampler.jl")
include("biomass_tools.jl")

outputpath(name) = normpath(joinpath("D:/Gabriel/SimOutput", "$name"))
data = load(outputpath("sims_30.jld2"))

function chain_moments(samps, cid::Int)
    web = samps["webs"][cid]
    αs = samps["αs"][cid]

    out = []
    for α in αs
        #TODO: need to seperate the moments of the αs into off diagonal: pos, neg, and diagonal
        #      as they all mean different stuff
        M = cmat(web, α)
        push!(out, [maximum(real.(eigvals(M))), mean(α), std(α), skewness(α), kurtosis(α)])
    end

    return out
end

out = chain_moments(data, 1)

M = cmat(data["webs"][1], data["αs"][1][1])
odiags = FoodWebs.offdiag_pairs(M)
[odiags[:, i] for i in 1:size(odiags, 1) if odiags[i, 1] != 0]

plt[:hist]([o[1] for o in out])
plt[:hist]([o[2] for o in out])
plt[:hist]([o[3] for o in out])
plt[:hist]([o[4] for o in out])
