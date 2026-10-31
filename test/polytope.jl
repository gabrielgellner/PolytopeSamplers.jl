using HRMCMC: Polytope, isfeasible

@testset "Polytope" begin
    triangle = Polytope([0.0, 0.0], [1.0, 1.0]; A = [1.0 1.0], b = [1.0])

    @testset "argument checks" begin
        @test_throws DimensionMismatch Polytope([0.0, 0.0], [1.0])
        @test_throws ArgumentError Polytope([0.0, 2.0], [1.0, 1.0])
        @test_throws DimensionMismatch Polytope([0.0, 0.0], [1.0, 1.0]; A = [1.0 1.0 1.0], b = [1.0])
        @test_throws DimensionMismatch Polytope([0.0, 0.0], [1.0, 1.0]; A = [1.0 1.0], b = [1.0, 2.0])
        @test_throws DimensionMismatch Polytope([0.0, 0.0], [1.0, 1.0]; C = [1.0 1.0], d = Float64[])
    end

    @testset "membership" begin
        @test [0.2, 0.3] in triangle
        @test [0.0, 1.0] in triangle
        @test !([0.6, 0.6] in triangle)
        @test !([-0.1, 0.5] in triangle)
        @test isfeasible(triangle, [0.6, 0.6]; atol = 0.25)
        line = Polytope([0.0, 0.0], [1.0, 1.0]; C = [1.0 -1.0], d = [0.0])
        @test [0.4, 0.4] in line
        @test !([0.4, 0.5] in line)
        @test_throws DimensionMismatch isfeasible(triangle, [0.1])
    end

    @testset "isempty" begin
        @test !isempty(triangle)
        @test isempty(Polytope([0.0, 0.0], [1.0, 1.0]; A = [-1.0 -1.0], b = [-3.0]))
        @test isempty(Polytope([0.0, 0.0], [1.0, 1.0]; C = [1.0 1.0], d = [5.0]))
        # Unbounded but not empty
        @test !isempty(Polytope([0.0], [Inf]))
    end

    @test sprint(show, triangle) == "Polytope in 2 variables with 1 inequalities and 0 equalities"
end
