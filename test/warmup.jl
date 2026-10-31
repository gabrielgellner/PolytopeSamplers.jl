using PolytopeSamplers: Polytope, Warmup, warmup
using LinearAlgebra: I, norm

@testset "warmup" begin
    @testset "errors" begin
        @test_throws ArgumentError warmup(Polytope([0.0, 0.0], [1.0, 1.0]; A = [-1.0 -1.0], b = [-3.0]))
        @test_throws ArgumentError warmup(Polytope([0.0, 0.0], [1.0, Inf]))
    end

    @testset "full-dimensional triangle" begin
        P = Polytope([0.0, 0.0], [1.0, 1.0]; A = [1.0 1.0], b = [1.0])
        w = warmup(P)
        @test w isa Warmup
        # The three vertices, each extreme in some coordinate
        @test sort(collect(eachcol(w.points))) == [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0]]
        @test w.center ≈ [1 / 3, 1 / 3]
        @test size(w.basis) == (2, 2)
        @test w.basis' * w.basis ≈ I
    end

    @testset "equalities restrict the directions" begin
        # The 3-simplex x ≥ 0, sum(x) = 1 in four variables
        P = Polytope(zeros(4), ones(4); C = ones(1, 4), d = [1.0])
        w = warmup(P)
        @test size(w.basis) == (4, 3)
        @test norm(ones(1, 4) * w.basis) < 1.0e-12
        @test sum(w.center) ≈ 1 atol = 1.0e-12
        @test all(p -> p in P, eachcol(w.points))
    end

    @testset "fixed coordinates have zero direction rows" begin
        P = Polytope([0.0, 0.5, 0.0], [1.0, 0.5, 1.0])
        w = warmup(P)
        @test size(w.basis) == (3, 2)
        @test all(iszero, w.basis[2, :])
        @test w.center[2] == 0.5
    end

    @testset "inequalities that hold with equality" begin
        # x1 + x2 ≤ 1 and x1 + x2 ≥ 1 leave the segment x1 + x2 = 1
        P = Polytope([0.0, 0.0], [1.0, 1.0]; A = [1.0 1.0; -1.0 -1.0], b = [1.0, -1.0])
        w = warmup(P)
        @test size(w.basis, 2) == 1
        @test abs(sum(w.basis)) < 1.0e-12
        # A coordinate pinned by inequalities, x2 ≤ 0.3 and x2 ≥ 0.3
        Q = Polytope([0.0, 0.0], [1.0, 1.0]; A = [0.0 1.0; 0.0 -1.0], b = [0.3, -0.3])
        wq = warmup(Q)
        @test size(wq.basis, 2) == 1
        @test all(iszero, wq.basis[2, :])
    end

    @testset "a single point" begin
        w = warmup(Polytope([0.2, 0.7], [0.2, 0.7]))
        @test size(w.basis) == (2, 0)
        @test w.center == [0.2, 0.7]
    end
end
