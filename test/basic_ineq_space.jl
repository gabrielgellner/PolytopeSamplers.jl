using LinearAlgebra
using Distributions
using RecursiveArrayTools
using PyPlot

# Given M α ≦ f:
M = [3/2 -1; 1 1; -2 -1]
f = [5.0, 10.0, -3.0]
upper = 5.5

# So it is easy to check if a point (α1, α2) statisfies M α ≦ f, we simply multiple and subtract
# and see if the result is negative.
#
# So for the first pass lets just do rejection sampling
chain = []
for i in 1:1e6
    α = rand(Uniform(0, 10), 2)
    if all(M * α - f .<= 0) && all(α .<= upper)
        push!(chain, α)
    end
end
chain = VectorOfArray(chain)

let
    plot(chain[1, :], chain[2, :], ",")
    xlim(-1, 10)
    ylim(-1, 10)
end
