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

is_feasible(prob, x; atol = 1e-15) = all(isapprox.(prob.E * x - prob.g, 0.0, atol = atol))

#TODO: if I call this more than once, I will get double the warm up points
# Try to build the warm up arrays
function warmup_sampler!(samp::ACHRSampler)
    # # Use JuMP to solve the linear programming problem
    m = Model(solver = ClpSolver())
    #TODO: figure out how to get the iterative lower and upper bounds like samp.prob.lower[i]
    @variable(m, samp.prob.lower[1] <= α[i = 1:size(samp.prob.E, 2)] <= samp.prob.upper[1])
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
    unique!(samp.warmup)
    #keep = np.logical_not(self._is_redundant(self.warmup))
    #self.warmup = self.warmup[keep, :]
    #self.n_warmup = self.warmup.shape[0]
    samp.center = mean(samp.warmup)
    return
end

"""
    Sample a new feasible point from the point `x` in direction `Δ`.
"""
function step(sampler, x, Δ)
    prob = sampler.prob
    lmin = -Inf
    lmax = Inf
    for i in 1:length(Δ)
        if Δ[i] > 0
            # scale minimum will satisfy: `x + scale * Δ ≥ lower` <=> `(lower - x) / Δ ≤ scale`
            scale = (prob.lower[i] - x[i]) / Δ[i]
            if scale > lmin
                lmin = scale
            end
        elseif Δ[i] < 0
            # scale maximum will satisfy `(lower - x) / Δ ≥ scale`
            scale = (prob.lower[i] - x[i]) / Δ[i]
            if scale < lmax
                lmax = scale
            end
        end
    end
    if lmin != lmax
        scale = rand(Uniform(lmin, lmax))
    else
        scale = lmin
    end
    return x + scale * Δ
end

#S = 10
#pnet = cascade_network(S, 0.7)
S = 3
pnet = PredationMatrix([0 1 1; 0 0 1; 0 0 0])
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

xout = [sample(sampler.warmup)]
for i in 1:100000
    Δ = sample(sampler.warmup) - sampler.center
    x_next = step(sampler, xout[end], Δ)
    push!(xout, copy(x_next))
end
xout
[is_feasible(sampler.prob, xout[i]) for i in 1:length(xout)]

using PyPlot
plot(xout[50000:end])
