module PolytopeSamplers

using AbstractMCMC: AbstractMCMC, AbstractMCMCEnsemble, AbstractModel, AbstractSampler,
    MCMCDistributed, MCMCSerial, MCMCThreads, sample
using HiGHS: HiGHS
using JuMP: @constraint, @objective, @variable, INFEASIBLE, INFEASIBLE_OR_UNBOUNDED,
    MAX_SENSE, MIN_SENSE, Model,
    is_solved_and_feasible, optimize!, set_lower_bound, set_silent, set_upper_bound,
    termination_status, value
using LinearAlgebra: I, Symmetric, cholesky, diag, dot, eigvals, issuccess, lu!, mul!, norm,
    normalize!, nullspace, pinv, svd
using Random: AbstractRNG, randexp, randn!
using MCMCDiagnosticTools: ess_rhat, mcse
using Statistics: mean, std

include("polytope.jl")
export Polytope, isfeasible

include("rounding.jl")
include("warmup.jl")
export warmup
public Warmup

include("sampler.jl")
include("billiard.jl")
export ACHR, BilliardWalk, CoordinateHitAndRun, HitAndRun
export MCMCDistributed, MCMCSerial, MCMCThreads, sample
public AbstractHitAndRun, PolytopeSampler

include("diagnostics.jl")
export diagnostics

end
