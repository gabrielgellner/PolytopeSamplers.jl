module PolytopeSamplersMCMCChainsExt

using AbstractMCMC: AbstractMCMC
using MCMCChains: Chains
using PolytopeSamplers: PolytopeSampler

# Return the points of a chain as an `MCMCChains.Chains`, with one parameter per coordinate
# of the polytope, named `x[1]`, `x[2]`, … unless `param_names` is given. AbstractMCMC
# combines the chains of an ensemble with `chainscat`.
function AbstractMCMC.bundle_samples(
        samples::Vector{Vector{Float64}}, ::AbstractMCMC.AbstractModel, ::PolytopeSampler, state,
        ::Type{Chains};
        param_names = nothing, stats = missing, discard_initial::Integer = 0, thinning::Integer = 1,
        kwargs...,
    )
    n = length(state.x)
    names = something(param_names, ["x[$i]" for i in 1:n])
    length(names) == n ||
        throw(DimensionMismatch("param_names has $(length(names)) names, the polytope has $n variables"))
    vals = Array{Float64}(undef, length(samples), n, 1)
    for (j, x) in enumerate(samples)
        vals[j, :, 1] = x
    end
    info = stats === missing ? (;) : (; start_time = stats.start, stop_time = stats.stop)
    return Chains(vals, names; start = discard_initial + 1, thin = thinning, info)
end

end
