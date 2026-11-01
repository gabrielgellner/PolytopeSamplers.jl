using PolytopeSamplers
using Test
using Aqua
using ExplicitImports: test_explicit_imports
using AbstractMCMC: AbstractMCMC

AbstractMCMC.setprogress!(false)

@testset "PolytopeSamplers.jl" begin
    @testset "Aqua" begin
        Aqua.test_all(PolytopeSamplers)
    end
    @testset "ExplicitImports" begin
        # HiGHS documents `HiGHS.Optimizer` as its JuMP entry point, and AbstractMCMC
        # documents `step`, `bundle_samples`, `getparams`, `getstats` and its abstract types as
        # the interface for samplers, but neither declares them `public`
        test_explicit_imports(
            PolytopeSamplers;
            all_explicit_imports_are_public = (;
                ignore = (:AbstractMCMCEnsemble, :AbstractModel, :AbstractSampler),
            ),
            all_qualified_accesses_are_public = (; ignore = (:Optimizer, :step, :bundle_samples, :getparams, :getstats)),
        )
    end
    include("polytope.jl")
    include("warmup.jl")
    include("sampler.jl")
    include("diagnostics.jl")
    include("correctness.jl")
    include("adaptive.jl")
    include("crhmc.jl")
    include("density.jl")
end
