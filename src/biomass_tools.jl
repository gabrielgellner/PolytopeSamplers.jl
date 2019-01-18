function is_pyramid(pnet, β)
    tls = trophic_levels(pnet)
    for i in 1:length(tls)
        for j in findall(tls[i] .< tls)
            # check all higher trophic levels for having higher biomass
            if β[i] < β[j]
                return false
            end
        end
    end
    return true
end

function is_inverted_pyramid(pnet, β)
    tls = trophic_levels(pnet)
    for i in 1:length(tls)
        for j in findall(tls[i] .< tls)
            # check all higher trophic levels for having lower biomass
            if β[i] > β[j]
                return false
            end
        end
    end
    return true
end

function set_β(pnet, δ, min_β)
    @assert min_β > 0
    # δ is the constant increase / decrease between whole trophic levels
    # so δ > 0 means top heavy,
    #    δ < 0 means eltonian

    tl_scale = FoodWebs.trophic_levels(pnet) .- 1

    if δ >= 0
        start = min_β
    elseif δ < 0
        # this ensures that:
        # 1. all biomass values will be positive
        # 2. the smallest value will be min_β
        start = min_β - maximum(tl_scale) * δ
    end
    return start .+ tl_scale .* δ
end
