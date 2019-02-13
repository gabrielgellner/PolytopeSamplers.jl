using LinearAlgebra
using Distributions
import JuMP
using Clp
using RecursiveArrayTools

abstract type AbstractMatrixConstraintProblem end
# This needs fields for E * b = g:
# biomass and growth
# - b::Vector{Float64}
# - g::Vector{Float64}
# box constraints
# - lower::Vector{Float64}
# - upper::Vector{Float64}
# equality coefficients
# - E::Matrix{Float64}

mutable struct ACHRSampler{T <: AbstractMatrixConstraintProblem}
    prob::T
    Z::Matrix{Float64} # nullspace of prob.E
    #bounds_tol? Or is this part of prob?
    warmup::Vector{Vector{Float64}}
    center::Vector{Float64}
    n_samples::Int # how many samples have already been generated, used in center calculations
    prev::Vector{Float64}
    #points?
end

is_feasible(prob, x; atol = 1e-15) = all(isapprox.(prob.E * x - prob.g, 0.0, atol = atol))

#TODO: if I call this more than once, I will get double the warm up points
function warmup_sampler!(samp::ACHRSampler; init = :center)
    # # Use JuMP to solve the linear programming problem
    m = JuMP.Model(solver = ClpSolver())
    #TODO: figure out how to get the iterative lower and upper bounds like samp.prob.lower[i]
    JuMP.@variable(m, samp.prob.lower[1] <= α[i = 1:size(samp.prob.E, 2)] <= samp.prob.upper[1])
    JuMP.@constraint(m, samp.prob.E * α - samp.prob.g .== 0)

    # Warm up along each individual dimension to get the Min and Max to form a spanning set
    for sense in (:Min, :Max)
        for i in 1:size(samp.prob.E, 2)
            JuMP.@objective(m, sense, α[i])
            status = JuMP.solve(m, suppress_warnings = true)
            if status == :Infeasible
                return :Infeasible
            end
            if status == :Optimal
                push!(samp.warmup, JuMP.getvalue(α))
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
    if init == :center
        samp.prev = copy(samp.center)
    elseif init == :bound
        samp.prev = copy(sample(samp.warmup))
    end
    return :Feasible
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
            @show "Hit and Run search direction is length 0"
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

    return samp.prev + scale * Δ
end

function update!(samp::ACHRSampler, Δ)
    samp.prev = step(samp, Δ)
    # update the center given the new iteration
    samp.center = ((samp.n_samples * samp.center) / (samp.n_samples + 1) + samp.prev / (samp.n_samples + 1))
    samp.n_samples += 1
end

"""
Generate a set of samples.

this is the basic sampling function for all hit-and-run samplers.

* Arguments
n::Int
    the number of samples that are generated at once

* Notes
Performance of this function linearly depends on the number of reactions in your model and the
thinning factor.
"""
function hr_sample!(samp::ACHRSampler, n::Int; thinning::Int = 1, burn_in::Int = -1)
    if burn_in == -1 # what is the correct way to do this in a typed manner?
        burn_in = round(Int, n / 2)
    end
    #TODO: use thinning + burn_in information to get this correct
    #chain = fill(0.0, n - burn_in, length(samp.center))
    chain = []

    for i in 1:n
        Δ = sample(samp.warmup) - samp.center
        update!(samp, Δ)
        # thin for memory saving
        if i % thinning == 0 && i > burn_in
            push!(chain, copy(samp.prev))
            #push!(cout, copy(samp.center))
        end
    end

    return VectorOfArray(chain)
end
