using LinearAlgebra
using FoodWebs
using DifferentialEquations
using RecursiveArrayTools
using PyPlot

include("sampler.jl")
include("biomass_tools.jl")

stability(M) = maximum(real.(eigvals(M)))
reactivity(M) = stability((M + M') / 2)
GLVpar(web, αs) = (A = α_mat(web, αs), web = web, M = cmat(web, αs))
fweb(u, p, t) = u .* (p.A * u - p.web.g)

function init_model(S, C; lb = 0.0, ub = 10.0, min_β = 10.0, δ = 0.0)
    pnet = cascade_network(S, C)
    β = set_β(pnet, δ, min_β)

    growth = fill(1.0, S)
    for s in basal_species(pnet)
        growth[s] = -1.0
    end
    return ConstrainedFoodWeb(pnet, β, growth, lb, ub)
end

S = 50
C = 0.23
δ = -1.0
min_β = 10.0

web = init_model(S, C, δ = δ, min_β = min_β)
sampler = ACHRSampler(web)

max_iter = 0
while true
    global web = init_model(S, C, δ = δ, min_β = min_β)
    global sampler = ACHRSampler(web)

    status = warmup_sampler!(sampler)
    if status == :Feasible
        break
    else
        max_iter += 1
    end

    if max_iter > 100
        error("could not find feasible start")
    end
end

chain = hr_sample!(sampler, 10000; burn_in = 0)

tspan = (0.0, 1000.0)
u0 = web.b + 0.5 * rand(S)
resilience = VectorOfArray([[stability(cmat(web, c)), reactivity(cmat(web, c))] for c in chain])
plot(resilience[1, :], resilience[2, :], "o")
#αs = chain[findmin(deigs)[2]]
stab_αs = [i for (i, r) in enumerate(resilience) if r[1] < 0]
αs = chain[sample(stab_αs)]

par = GLVpar(web, αs)

prob = ODEProblem(fweb, u0, tspan, par)
sol = solve(prob, Rosenbrock23(), abstol = 1e-6)

let
    tvals = range(tspan[1], sol.t[end], length = 10000)
    plot(tvals, sol(tvals).u)
    title("Re(λ_max) = $(stability(cmat(web, αs)))\n v0 = $(reactivity(cmat(web, αs)))")
end
