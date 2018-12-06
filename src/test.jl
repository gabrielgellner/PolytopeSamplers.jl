using JuMP
using Clp
using FoodWebs
using Statistics
using StatsBase
include("foodweb_utils.jl")

# Try to build the warm up arrays
function warmup_sampler(model::ConstrainedFoodWeb)
    # # Use JuMP to solve the linear programming problem
    m = Model(solver = ClpSolver())
    @variable(m, model.lower[i] <= α[i = 1:size(model.E, 2)] <= model.upper[i])
    @constraint(m, model.E * α - model.g .== 0)

    # Warm up along each individual dimension to get the Min and Max to form a spanning set
    warmup = []
    for sense in (:Min, :Max)
        for i in 1:size(model.E, 2)
            @objective(m, sense, α[i])
            status = solve(m) #, suppress_warnings = true)
            if status == :Optimal
                push!(warmup, getvalue(α))
            end
        end
    end
    return warmup
end

function step(sampler, x, Δ, fraction = nothing, tries = 0)

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
@time warmup = warmup_sampler(model)

center = mean(warmup)
Δ = sample(warmup) - center
