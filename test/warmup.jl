using PolytopeSamplers: Polytope, Warmup, warmup
using LinearAlgebra: I, norm, svdvals
using PolytopeSamplers: PolytopeSamplers

@testset "warmup" begin
    @testset "errors" begin
        @test_throws ArgumentError warmup(Polytope([0.0, 0.0], [1.0, 1.0]; A = [-1.0 -1.0], b = [-3.0]))
        @test_throws ArgumentError warmup(Polytope([0.0, 0.0], [1.0, Inf]))
    end

    @testset "full-dimensional triangle" begin
        P = Polytope([0.0, 0.0], [1.0, 1.0]; A = [1.0 1.0], b = [1.0])
        w = warmup(P; rounding = false)
        @test w isa Warmup
        @test !w.rounded
        # The three vertices, each extreme in some coordinate
        @test sort(collect(eachcol(w.points))) == [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0]]
        @test w.center ≈ [1 / 3, 1 / 3]
        @test size(w.transform) == (2, 2)
        @test w.transform' * w.transform ≈ I
        # The reduced constraints are the four bounds and the diagonal edge, with unit-norm
        # rows
        @test size(w.G) == (5, 2)
        @test all(r -> norm(r) ≈ 1, eachrow(w.G))
        @test w.zpoints ≈ w.transform' * (w.points .- w.center)
        # The maximum-volume ellipse of a triangle is centered at its centroid
        @test warmup(P).center ≈ [1 / 3, 1 / 3] atol = 1.0e-4
    end

    @testset "rounding" begin
        # The box [0, 100] × [0, 1] rounds to a square: the inscribed ellipse has semi-axes
        # 50 and 1/2 and every constraint lies at distance one from the center
        w = warmup(Polytope([0.0, 0.0], [100.0, 1.0]))
        @test w.rounded
        @test w.center ≈ [50, 0.5] rtol = 1.0e-4
        @test sort(svdvals(w.transform)) ≈ [0.5, 50] rtol = 1.0e-3
        @test w.h ≈ ones(4) rtol = 1.0e-4
        # A long thin simplex: rounding leaves a polytope whose largest inscribed ball has
        # radius near that of the regular simplex in the same dimension
        P = Polytope(zeros(3), [1000.0, 1.0, 1.0]; A = [0.001 1.0 1.0], b = [1.0])
        w = warmup(P)
        @test w.rounded
        @test all(x -> x in P, eachcol(w.transform * w.zpoints .+ w.center))
        _, r = PolytopeSamplers.chebyshev_center(w.G, w.h, PolytopeSamplers.HiGHS.Optimizer)
        @test 0.9 < r < 1.1
    end

    @testset "maximum-volume ellipsoid of a box" begin
        A = [1.0 0.0; -1.0 0.0; 0.0 1.0; 0.0 -1.0]
        b = [3.0, 1.0, 1.0, 1.0] # [-1, 3] × [-1, 1]
        E2, c, converged = PolytopeSamplers.max_volume_ellipsoid(A, b, [0.5, 0.1]; reg = 1.0e-10)
        @test converged
        @test c ≈ [1, 0] atol = 1.0e-5
        # Semi-axes 2 and 1: E2 = diag(1/4, 1)
        @test E2 ≈ [0.25 0; 0 1] atol = 1.0e-4
        @test_throws ArgumentError PolytopeSamplers.max_volume_ellipsoid(A, b, [5.0, 0.0]; reg = 1.0e-3)
    end

    @testset "equalities restrict the directions" begin
        # The 3-simplex x ≥ 0, sum(x) = 1 in four variables
        P = Polytope(zeros(4), ones(4); C = ones(1, 4), d = [1.0])
        w = warmup(P)
        @test size(w.transform) == (4, 3)
        @test norm(ones(1, 4) * w.transform) < 1.0e-12
        @test sum(w.center) ≈ 1 atol = 1.0e-12
        @test all(p -> p in P, eachcol(w.points))
    end

    @testset "fixed coordinates have zero direction rows" begin
        P = Polytope([0.0, 0.5, 0.0], [1.0, 0.5, 1.0])
        w = warmup(P)
        @test size(w.transform) == (3, 2)
        @test all(iszero, w.transform[2, :])
        @test w.center[2] == 0.5
    end

    @testset "inequalities that hold with equality" begin
        # x1 + x2 ≤ 1 and x1 + x2 ≥ 1 leave the segment x1 + x2 = 1
        P = Polytope([0.0, 0.0], [1.0, 1.0]; A = [1.0 1.0; -1.0 -1.0], b = [1.0, -1.0])
        w = warmup(P)
        @test size(w.transform, 2) == 1
        @test abs(sum(w.transform)) < 1.0e-12
        # A coordinate pinned by inequalities, x2 ≤ 0.3 and x2 ≥ 0.3
        Q = Polytope([0.0, 0.0], [1.0, 1.0]; A = [0.0 1.0; 0.0 -1.0], b = [0.3, -0.3])
        wq = warmup(Q)
        @test size(wq.transform, 2) == 1
        @test all(iszero, wq.transform[2, :])
    end

    @testset "a single point" begin
        w = warmup(Polytope([0.2, 0.7], [0.2, 0.7]))
        @test size(w.transform) == (2, 0)
        @test w.center == [0.2, 0.7]
    end
end
