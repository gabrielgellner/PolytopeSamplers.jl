module HRMCMC

using HiGHS: HiGHS
using JuMP: @constraint, @objective, @variable, INFEASIBLE, INFEASIBLE_OR_UNBOUNDED,
    MAX_SENSE, MIN_SENSE, Model,
    is_solved_and_feasible, optimize!, set_lower_bound, set_silent, set_upper_bound,
    termination_status, value
using LinearAlgebra: I, mul!, norm, normalize!, nullspace, pinv, svd
using Random: AbstractRNG, default_rng, randn!
using Statistics: mean

include("polytope.jl")
export Polytope, isfeasible

include("warmup.jl")
export warmup
public Warmup

include("sampler.jl")
export ACHR, HitAndRun, hr_sample
public AbstractHitAndRun

end
