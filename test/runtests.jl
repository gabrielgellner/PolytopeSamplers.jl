using PolytopeSamplers
using Test
using Aqua
using ExplicitImports: test_explicit_imports

@testset "PolytopeSamplers.jl" begin
    @testset "Aqua" begin
        Aqua.test_all(PolytopeSamplers)
    end
    @testset "ExplicitImports" begin
        # HiGHS documents `HiGHS.Optimizer` as its JuMP entry point but does not declare it
        # `public`
        test_explicit_imports(
            PolytopeSamplers;
            all_qualified_accesses_are_public = (; ignore = (:Optimizer,)),
        )
    end
    include("polytope.jl")
    include("warmup.jl")
    include("sampler.jl")
end
