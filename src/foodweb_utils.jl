import FoodWebs: PredationMatrix, top_species

#TODO: simplify these utilities to be more direct
function pmat2cnet(pmat::PredationMatrix)
    adj = -pmat + pmat'
    for i in 1:pmat.S
        adj[i, i] = -1
    end
    return adj
end

function cnet_nz(cnet::Matrix)
    α_nz = [[] for i in 1:size(cnet, 1)]
    for i in 1:size(cnet, 1)
        for j in 1:size(cnet, 2)
            if cnet[i, j] != 0
                push!(α_nz[i], (i, j))
            end
        end
    end
    return α_nz
end

function cnet2E(cnet::Matrix, b::Vector)
    S = size(cnet, 1)
    @assert S == length(b)

    α_nz = cnet_nz(cnet)
    np = sum(length.(α_nz))
    E = fill(0.0, S, np)
    conmat = cnet .* b'

    last_end = length(α_nz[1])
    E[1, 1:last_end] = [conmat[αidx...] for αidx in α_nz[1]]
    for (i, α) in enumerate(α_nz[2:end])
        cur_end = last_end + length(α)
        E[i + 1, (last_end + 1):(cur_end)] = [conmat[αidx...] for αidx in α]
        last_end = cur_end
    end
    return E
end

struct ConstrainedFoodWeb
    # Network
    net::Matrix{Int}
    # biomass and growth
    b::Vector{Float64}
    g::Vector{Float64}
    # box constraints
    lower::Vector{Float64}
    upper::Vector{Float64}
    # equality coefficients
    E::Matrix{Float64}
end

function ConstrainedFoodWeb(pmat, b::Vector{T}, g::Vector{T}, lower::T, upper::T) where T
    net = pmat2cnet(pmat)
    for s in top_species(pmat)
        net[s, s] = 0
    end
    E = cnet2E(net, b)
    vlower = fill(lower, size(E, 2))
    vupper = fill(upper, size(E, 2))
    @assert length(b) == length(g) == pmat.S
    @assert length(vlower) == length(vupper) == size(E, 2)
    return ConstrainedFoodWeb(net, b, g, vlower, vupper, E)
end
