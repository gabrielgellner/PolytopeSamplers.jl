using FoodWebs
using MCMCDiagnostics
using PyPlot

include("sampler.jl")

S = 5
pnet = cascade_network(S, 0.8)
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
chain1 = hr_sample!(sampler, 1_000_000; burn_in = 0)

sampler = ACHRSampler(model)
@time warmup_sampler!(sampler, init = :bound)
chain2 = hr_sample!(sampler, 1_000_000; burn_in = 0)

sampler = ACHRSampler(model)
@time warmup_sampler!(sampler, init = :bound)
chain3 = hr_sample!(sampler, 1_000_000; burn_in = 0)

plot(chain1[5, (end - 10000):end])
plot(chain2[5, (end - 10000):end])
plot(chain3[5, (end - 10000):end])

effective_sample_size(chain1[5, :])

for i in 10:10:5000
    @show i, potential_scale_reduction(chain1[5, :][1:i], chain2[5, :][1:i], chain3[5, :][1:i])
end
