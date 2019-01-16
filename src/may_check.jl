using LinearAlgebra
using FoodWebs
using MCMCDiagnostics
using PyPlot

include("sampler.jl")
include("biomass_tools.jl")

function set_β(pnet, ξ)
    # Set the b vector according to the TP so have E[b_i] = ξTP + (1 - ξ),
    # so ξ > 0 means top heavy,
    #    ξ < 0 means eltonian
    # where ξ = biomass (the single input value)
    #
    # not sure I need this randomness
    # (ξ .* FoodWebs.trophic_levels(pnet) .+ (1 - ξ)) .* rand(Uniform(0, 1), S)
    return ξ .* FoodWebs.trophic_levels(pnet) .+ (1 - ξ)
end

function init_model(S, C; lb = 0.0, ub = 10.0, biomass = nothing)
    pnet = cascade_network(S, C)
    if biomass == nothing
        biomass = rand(S)
    elseif length(biomass) == 1
        β = set_β(biomass)
    end
    growth = fill(1.0, S)
    for s in basal_species(pnet)
        growth[s] = -1.0
    end
    return ConstrainedFoodWeb(pnet, β, growth, lb, ub)
end

function same_S(S, C; nsamp = 10, slope = 0.0)
    out = []
    @progress for i in 1:nsamp
        #biomass = rand(S)
        biomass = slope
        #TODO: this preallocating to avoid scope issues in the while feels bad
        web = init_model(S, C, biomass = biomass)
        sampler = ACHRSampler(web)

        max_iter = 0
        while true
            web = init_model(S, C, biomass = biomass)
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

        chain = hr_sample!(sampler, 10_000; burn_in = 0)

        push!(out, mean([maximum(real.(eigvals(cmat(web, c)))) for c in chain[(end - 1000):end]]))
    end

    return out
end

function div_stab(slope)
    out = []
    C = 0.25
    @progress for S in 10:30
        push!(out, [S, mean(same_S(S, C, nsamp = 60, slope = slope))])
    end

    return VectorOfArray(out)
end

out = []
for slope in range(-2.0, stop = 2.0, length = 5)
    @time push!(out, div_stab(slope))
end

let
    for o in out
        plot(o[1, :], o[2, :], "o")
    end
    xlabel("Diversity (S)")
    ylabel(L"Re(\lambda_{max})")
end


function check_stab()
    out = []
    S = 20
    C = 0.25
    @progress for slope in range(-3, stop = 5, length = 30)
        push!(out, [slope, mean(same_S(S, C, nsamp = 60, slope = slope))])
    end

    return VectorOfArray(out)
end

@time out = check_stab()

let
    plot(out[1, :], out[2, :])
    xlabel("slope")
    ylabel("mean stability")
end

pnet = cascade_network(20, 0.25)


set_β(pnet, -0.5)


pnet = cascade_network(30, 0.25)
σ = 0.5
bscale = σ .* FoodWebs.trophic_levels(pnet) .+ (1 - σ)
biomass = bscale #.* rand(Uniform(0, 1), pnet.S)

is_pyramid(pnet, biomass)
is_inverted_pyramid(pnet, biomass)
