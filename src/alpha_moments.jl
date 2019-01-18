using LinearAlgebra
using FoodWebs
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

function resample_stab(S, C; δ = 0.0, min_β = 10.0, nsamp = 200)
    out = []
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

        chain = hr_sample!(sampler, 200_000; burn_in = 0)
        deigs = [maximum(real.(eigvals(cmat(web, c)))) for c in chain[(end - 10000):end]]
        μ_α = mean(mean.(chain[(end - 10000):end]))
        push!(out, [S, mean(deigs), μ_α])
    end

    out = VectorOfArray(out)
end

# # Get a sense of how much sampling is required for a good mean stability
@time out = resample_stab(60, 0.25; δ = -1.0, nsamp = 100)
plot(1:length(out[2, :]), cumsum(out[2, :]) ./ collect(1:length(out[2, :])))
plot(1:length(out[3, :]), cumsum(out[3, :]) ./ collect(1:length(out[3, :])))

# # Build a Diversity-Stability result and calculate the variance around the mean stability
out = []
@time @progress for S in 10:60
    push!(out, resample_stab(S, 0.25; δ = -1.0, nsamp = 300))
end

#TODO: this code is ugly, but works
data = fill(0.0, 4, length(out))
for i in 1:length(out)
    raw = [o[2] for o in out[i]]
    data[1, i] = mean(raw)
    data[2, i] = std(raw)
    data[3, i] = mean([o[3] for o in out[i]])
    data[4, i] = std([o[3] for o in out[i]])
end

let
    figure()
    plot([o[1][1] for o in out], data[1, :])
    plot([o[1][1] for o in out], data[1, :] - data[2, :], "k")
    plot([o[1][1] for o in out], data[1, :] + data[2, :], "k")
    ylabel("Stability")

    figure()
    plot([o[1][1] for o in out], data[3, :])
    plot([o[1][1] for o in out], data[3, :] - data[4, :], "k")
    plot([o[1][1] for o in out], data[3, :] + data[4, :], "k")
    ylabel("Alpha")
end
