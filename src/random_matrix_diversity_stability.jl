using LinearAlgebra
using Distributions
using FoodWebs
using PyPlot

function div_stab(S, C)
    # this gives a parabola for alpha mean about what the constrained webs are
    a = -2.5 / (20^2 - 80)
    b = -40 * a
    # mean_alpha ≈ a * S ^ 2 + b * S

    resamp = 1000
    deigs = []
    for i in 1:200
        pnet = cascade_network(S, C)
        eig_sum = 0
        for rs in 1:resamp
            cmat = random_predprey(pnet, Uniform(0, 2 * (a * S ^ 2 + b * S)), 0.8)
            eig_sum += λ_stability(cmat)
        end
        # and the mean
        push!(deigs, eig_sum / resamp)
    end
    return deigs
end

out = []
@progress for S in 10:30
    push!(out, div_stab(S, 0.25))
end

#TODO: this code is ugly, but works
data = fill(0.0, 2, length(out))
for i in 1:length(out)
    data[1, i] = mean(out[i])
    data[2, i] = std(out[i])
end

let
    Svals = collect(10:30)
    plot(Svals, data[1, :])
    plot(Svals, data[1, :] - data[2, :], "k")
    plot(Svals, data[1, :] + data[2, :], "k")
end


Svals = 10:30
a = -2.5 / (20^2 - 80)
b = -40 * a

out = []
for S in Svals
    dist = Uniform(0, 2 * (a * S ^ 2 + b * S))
    push!(out, [S, mean(dist), std(dist)])
end
out = VectorOfArray(out)

let
    plot(out[1, :], out[2, :])
    plot(out[1, :], out[2, :] + out[3, :])
    plot(out[1, :], out[2, :] - out[3, :])
end
