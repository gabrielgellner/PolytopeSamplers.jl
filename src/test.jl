using JuMP
using Clp
using FoodWebs
using LinearAlgebra
using Statistics
using StatsBase
include("foodweb_utils.jl")

struct ACHRSampler
    prob::ConstrainedFoodWeb
    null_basis::Matrix{Float64}
    #bounds_tol? Or is this part of prob?
    warmup::Vector{Vector{Float64}}
    #cur?
    #points?
end

ACHRSampler(prob::ConstrainedFoodWeb) = ACHRSampler(prob, nullspace(prob.E), [])

# Try to build the warm up arrays
function warmup_sampler!(samp::ACHRSampler)
    # # Use JuMP to solve the linear programming problem
    m = Model(solver = ClpSolver())
    @variable(m, samp.prob.lower[i] <= α[i = 1:size(samp.prob.E, 2)] <= samp.prob.upper[i])
    @constraint(m, samp.prob.E * α - samp.prob.g .== 0)

    # Warm up along each individual dimension to get the Min and Max to form a spanning set
    #warmup = []
    for sense in (:Min, :Max)
        for i in 1:size(samp.prob.E, 2)
            @objective(m, sense, α[i])
            status = solve(m) #, suppress_warnings = true)
            if status == :Optimal
                push!(samp.warmup, getvalue(α))
            end
        end
    end
    return
end

"""
    Sample a new feasible point from the point `x` in direction `Δ`.
"""
function step(sampler, x, Δ, fraction = nothing, tries = 0)
    prob = sampler.problem
    # permissible scalings for staying in variable bounds
    vscale = (1 .- bounds_tol) .* (prob.variable_bounds .- x)
    vscale = flatten(vscale ./ Δ)

    scale = rand(Uniform(vscale[1], vscale[2]))
    next_x = x + scale * Δ

    return next_x
end


S = 20
pnet = cascade_network(S, 0.25)
biomass = rand(S)
growth = fill(1.0, S)
for s in basal_species(pnet)
    growth[s] = -1.0
end
model = ConstrainedFoodWeb(pnet, biomass, growth, 0.0, 10.0)
size(model.E, 2)

sampler = ACHRSampler(model)
@time warmup_sampler!(sampler)
sampler.warmup
center = mean(sampler.warmup)
Δ = sample(sampler.warmup) - center
