using HRMCMC
using Test
using Aqua
using ExplicitImports: test_explicit_imports

@testset "HRMCMC.jl" begin
    @testset "Aqua" begin
        Aqua.test_all(HRMCMC)
    end
    @testset "ExplicitImports" begin
        # HiGHS documents `HiGHS.Optimizer` as its JuMP entry point but does not declare it
        # `public`
        test_explicit_imports(
            HRMCMC;
            all_qualified_accesses_are_public = (; ignore = (:Optimizer,)),
        )
    end
    include("polytope.jl")
    include("warmup.jl")
    include("sampler.jl")
end
