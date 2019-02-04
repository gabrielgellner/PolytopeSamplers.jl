using LinearAlgebra
using JLD2
using FileIO
using FoodWebs
using PyPlot

include("sampler.jl")
include("biomass_tools.jl")

#outputpath(name) = normpath(joinpath("C:/Users/gabri", "LargeOutput/$name"))
outputpath(name) = normpath(joinpath("D:/Gabriel/SimOutput", "$name"))

data = load(outputpath("sims_30.jld2"))

function link_dist(web, αs)
    M = cmat(web, αs)

    adj = Dict(i => (pos = Float64[], neg = Float64[]) for i in 1:size(M, 2))
    for j in 1:size(M, 2)
        for i in 1:size(M, 1)
            if i != j && M[i, j] < 0
                push!(adj[j].neg, M[i, j])
                push!(adj[j].pos, M[j, i])
            end
        end
    end
    for (k, v) in adj
        if length(v.pos) == 0
            delete!(adj, k)
        end
    end
    return adj
end

mean_IS = Dict()
ldist = link_dist(data["webs"][5], data["αs"][5][1])
for (ispecies, IS) in ldist
    mean_IS[length(IS.neg)] = [mean(IS.pos), mean(IS.neg)]
end
for ichain in 2:length(data["αs"][5])
    ldist = link_dist(data["webs"][5], data["αs"][5][ichain])
    for (ispecies, IS) in ldist
        mean_IS[length(IS.neg)][1] = (mean_IS[length(IS.neg)][1] + mean(IS.pos)) / 2
        mean_IS[length(IS.neg)][2] = (mean_IS[length(IS.neg)][2] + mean(IS.neg)) / 2
    end
end

let
    figure()
    for (nlink, IS) in mean_IS
        plot(nlink, IS[1], "ok")
    end

    figure()
    for (nlink, IS) in mean_IS
        plot(nlink, IS[2], "ok")
    end

    figure()
    for (nlink, IS) in mean_IS
        plot(nlink, sqrt(abs(IS[1]) * abs(IS[2])), "ok")
    end
    title("root mean")
end
