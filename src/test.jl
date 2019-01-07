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
    n_samples::Int # how many samples have already been generated, used in center calculations
    prev::Vector{Float64}
    #points?
end

ACHRSampler(prob::ConstrainedFoodWeb) = ACHRSampler(prob, nullspace(prob.E), [], fill(NaN, size(prob.E, 2)), 0, fill(NaN, size(prob.E, 2)))

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
    samp.n_samples = length(samp.warmup)
    samp.center = mean(samp.warmup)
    # start at the current estimated center
    samp.prev = copy(samp.center)
    return
end

"""
    Sample a new feasible point from the point `x` in direction `Δ`.
"""
function step(samp, Δ)
    prob = samp.prob
    lmin = -Inf
    lmax = Inf
    for i in 1:length(Δ)
        if Δ[i] ≈ 0
            @show "delta 0"
            continue
        end
        scale_low = (prob.lower[i] - samp.prev[i]) / Δ[i]
        scale_upp = (prob.upper[i] - samp.prev[i]) / Δ[i]
        if Δ[i] > 0
            # scale will satisfy: `(low - prev) / Δ <= scale <= (upp - prev) / Δ`
            if scale_low > lmin
                lmin = scale_low
            end
            if scale_upp < lmax
                lmax = scale_upp
            end
        elseif Δ[i] < 0
            # scale will satisfy: `(low - prev) / Δ >= scale >= (upp - prev) / Δ`
            if scale_low < lmax
                lmax = scale_low
            end
            if scale_upp > lmin
                lmin = scale_upp
            end
        end
    end
    if lmin != lmax
        scale = rand(Uniform(lmin, lmax))
    else
        scale = lmin
    end
    #@show (lmin, lmax)
    #@show samp.prev + lmin * Δ
    #@show samp.prev + lmax * Δ
    next = samp.prev + scale * Δ
    #@show is_feasible(samp.prob, next)
    return next
end

function update!(samp::ACHRSampler, Δ)
    samp.prev = step(samp, Δ)
    # update the center given the new iteration
    samp.center = ((samp.n_samples * samp.center) / (samp.n_samples + 1) + samp.prev / (samp.n_samples + 1))
    samp.n_samples += 1
end

S = 50
pnet = cascade_network(S, 0.3)
#S = 3
#pnet = PredationMatrix([0 1 1; 0 0 1; 0 0 0])
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

xout = [copy(sampler.prev)]
@time for i in 1:1_000_000
    Δ = sample(sampler.warmup) - sampler.center
    update!(sampler, Δ)
    push!(xout, copy(sampler.prev))
end
xout
sampler.center
using PyPlot
xdata = hcat(xout...)'
plot(xdata[(end - 500000):100:end, 1])


[is_feasible(sampler.prob, xout[i]) for i in 1:length(xout)]
Δ = sample(sampler.warmup) - sampler.center
step(sampler, Δ)
