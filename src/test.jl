using JuMP
using Clp
using FoodWebs
using LinearAlgebra
using Distributions
using Statistics
using StatsBase
include("foodweb_utils.jl")

mutable struct ACHRSampler
    prob::ConstrainedFoodWeb
    Z::Matrix{Float64} # nullspace of prob.E
    #bounds_tol? Or is this part of prob?
    warmup::Vector{Vector{Float64}}
    center::Vector{Float64}
    #cur?
    #points?
end

ACHRSampler(prob::ConstrainedFoodWeb) = ACHRSampler(prob, nullspace(prob.E), [], fill(NaN, size(prob.E, 2)))

#TODO: if I call this more than once, I will get double the warm up points
# Try to build the warm up arrays
function warmup_sampler!(samp::ACHRSampler)
    # # Use JuMP to solve the linear programming problem
    m = Model(solver = ClpSolver())
    @variable(m, samp.prob.lower[i] <= α[i = 1:size(samp.prob.E, 2)] <= samp.prob.upper[i])
    @constraint(m, samp.prob.E * α - samp.prob.g .== 0)

    # Warm up along each individual dimension to get the Min and Max to form a spanning set
    for sense in (:Min, :Max)
        for i in 1:size(samp.prob.E, 2)
            @objective(m, sense, α[i])
            status = solve(m) #, suppress_warnings = true)
            if status == :Optimal
                push!(samp.warmup, getvalue(α))
            end
        end
    end
    #TODO: Remove redundent search directions
    #keep = np.logical_not(self._is_redundant(self.warmup))
    #self.warmup = self.warmup[keep, :]
    #self.n_warmup = self.warmup.shape[0]
    #samp.center = mean(samp.warmup)
    return
end

"""
    Sample a new feasible point from the point `x` in direction `Δ`.
"""
function step(sampler, x, Δ)
    prob = sampler.prob
    # permissible scalings for staying in variable bounds
    vscale = vcat((prob.lower .- x) ./ Δ,
                  (prob.upper .- x) ./ Δ)
    pos_scale = vscale[vscale .> 0]
    neg_scale = vscale[vscale .<= 0]
    if length(pos_scale) > 0
        max_neg = maxium(neg_scale)
    else
        max_neg = 0.0
    end
    if length(neg_scale) > 0
        min_pos = maxium(neg_scale)
    else
        min_pos = 0.0
    end
    if min_pos != 0 && max_neg != 0
        scale = rand(Uniform(min_pos, max_neg))
    else
        scale = 0.0
    end

    return x + scale * Δ
end


S = 20
pnet = cascade_network(S, 0.25)
biomass = rand(S)
growth = fill(1.0, S)
for s in basal_species(pnet)
    growth[s] = -1.0
end
model = ConstrainedFoodWeb(pnet, biomass, growth, 0.0, 10.0)
#TODO: should I have length(prob) == size(prob.E, 2) ?
size(model.E, 2)

sampler = ACHRSampler(model)
@time warmup_sampler!(sampler)

sampler.warmup
sampler.center
Δ = sample(sampler.warmup) - sampler.center

x = sample(sampler.warmup)

step(sampler, x, Δ)
