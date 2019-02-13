using LinearAlgebra
using FoodWebs
using RecursiveArrayTools
using PyPlot

include("sampler.jl")
include("biomass_tools.jl")

function init_model(S, C; lb = 0.0, ub = 10.0, min_β = 10.0, δ = -1.0)
    pnet = generalized_cascade_network(S, C)
    β = set_β(pnet, δ, min_β)

    growth = fill(1.0, S)
    for s in basal_species(pnet)
        growth[s] = -1.0
    end
    return ConstrainedFoodWeb(pnet, β, growth, lb, ub)
end

function offdiag_pairs_nz(M)
    out = []
    # Look at the upper triangle
    for i in 1:(size(M, 1) - 1)
        for j in (i + 1):size(M, 2)
            if M[i, j] != 0
                push!(out, [M[i, j], M[j, i]])
            end
        end
    end
    return VectorOfArray(out)
end

function cmat_summary(web, αs)
    nzpairs = offdiag_pairs_nz(α_mat(web, αs))

    return (λ_1 = λ_stability(cmat(web, αs)),
            μ_α_neg = mean(nzpairs[1, :]),
            μ_α_pos = mean(nzpairs[2, :]),
            skew_α_neg = skewness((nzpairs[1, :])),
            skew_α_pos = skewness((nzpairs[1, :])))
end

function chain_summary(web, chain)
    out = []
    for c in chain
        push!(out, cmat_summary(web, c))
    end
    return (λ_1 = mean(d.λ_1 for d in out), μ_α_neg = mean(d.μ_α_neg for d in out), μ_α_pos = mean(d.μ_α_pos for d in out), skew_α_neg = mean(d.skew_α_neg for d in out), skew_α_pos = mean(d.skew_α_pos for d in out))
end

function chain_details(web, chain)
    out = []
    for c in chain
        push!(out, cmat_summary(web, c))
    end
    return out
end

function run_sampler(S, C; δ = -1.0, min_β = 10.0, nsteps = 10000)
    web = init_model(S, C, δ = δ, min_β = min_β)
    sampler = ACHRSampler(web)

    max_iter = 0
    while true
        web = init_model(S, C, δ = δ, min_β = min_β)
        sampler = ACHRSampler(web)

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

    chain = hr_sample!(sampler, nsteps; burn_in = 0)
    return (web = web, chain = chain)
end

function web_summary(S, C; δ = -1.0, min_β = 10.0, nsteps = 10000)
    web = init_model(S, C, δ = δ, min_β = min_β)
    sampler = ACHRSampler(web)

    max_iter = 0
    while true
        web = init_model(S, C, δ = δ, min_β = min_β)
        sampler = ACHRSampler(web)

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

    chain = hr_sample!(sampler, nsteps; burn_in = 0)

    return chain_summary(web, chain)
end

Svals = 10:30
out = []
@progress for S in Svals
    reps = [web_summary(S, 0.25; nsteps = 10_000) for i in 1:50]
    push!(out, (λ_1 = mean(r.λ_1 for r in reps), μ_α_pos = mean(r.μ_α_pos for r in reps), μ_α_neg = mean(r.μ_α_neg for r in reps), skew_α_neg = mean(r.skew_α_neg for r in reps), skew_α_pos = mean(r.skew_α_pos for r in reps)))
end

let
    figure()
    plot(Svals, [o.λ_1 for o in out], "o")
    xlabel("S")
    ylabel("λ_1")

    figure()
    plot(Svals, [o.μ_α_pos for o in out], "o")
    xlabel("S")
    ylabel("Mean α pos")

    figure()
    plot(Svals, [o.μ_α_neg for o in out], "o")
    xlabel("S")
    ylabel("Mean α neg")

    figure()
    plot(Svals, [o.skew_α_pos for o in out], "o")
    xlabel("S")
    ylabel("Skewness α pos")

    figure()
    plot(Svals, [o.skew_α_neg for o in out], "o")
    xlabel("S")
    ylabel("Skewness α neg")
end

out = []
@progress for i in 1:50
    nsteps = 100_000
    burning = round(Int, nsteps / 2)
    thin = 1
    samp = run_sampler(30, 0.25; nsteps = nsteps)
    push!(out, chain_summary(samp.web, samp.chain))
end

let
    plot([o.λ_1 for o in out], [o.μ_α_neg for o in out], "o")
    xlim(-0.5, 0.5)
end

let
    plot([o.λ_1 for o in out], [o.μ_α_pos for o in out], "o")
    xlim(-0.5, 0.5)
end


for i in 1:8
    nsteps = 100_000
    burnin = round(Int, nsteps / 2)
    thin = 10
    samp = run_sampler(30, 0.25; nsteps = nsteps)
    data1 = chain_details(samp.web, samp.chain)[burnin:thin:end]
    plot([d.λ_1 for d in data1 if d.λ_1 < 0], [d.μ_α_neg for d in data1 if d.λ_1 < 0], "o", mfc = "none")
    #plot([d.λ_1 for d in data1 if d.λ_1 < 0], [d.μ_α_pos for d in data1 if d.λ_1 < 0], "o", mfc = "none")
end

for i in 1:5
    nsteps = 10_000
    burnin = round(Int, nsteps / 2)
    thin = 2
    samp = run_sampler(30, 0.25; nsteps = nsteps)
    data1 = chain_details(samp.web, samp.chain)[burnin:thin:end]
    plot([d.μ_α_pos for d in data1 if d.λ_1 < 0], [d.μ_α_neg for d in data1 if d.λ_1 < 0], "o", mfc = "none")
end
