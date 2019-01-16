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
