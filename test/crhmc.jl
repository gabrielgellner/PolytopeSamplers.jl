using PolytopeSamplers: PolytopeSamplers, CRHMC, Polytope, warmup
using LinearAlgebra: norm
using Random: Xoshiro, randn

@testset "CRHMC" begin
    rng = Xoshiro(3)
    k = 6
    A = randn(rng, 20, k)
    P = Polytope(fill(-3.0, k), fill(3.0, k); A, b = ones(20) .+ rand(rng, 20))
    w = warmup(P)

    @testset "gradients match finite differences" begin
        c = PolytopeSamplers.chain_state(CRHMC(), w)
        z = 0.1 .* randn(rng, k)
        p = randn(rng, k)
        fresh() = PolytopeSamplers.chain_state(CRHMC(), w)
        U(z) = (s = fresh(); PolytopeSamplers.metric!(s, z); s.logdet)
        K(z, p) = PolytopeSamplers.hamiltonian(fresh(), z, p) - U(z)
        e(i) = (v = zeros(k); v[i] = 1; v)
        ε = 1.0e-6
        gU = zeros(k)
        @test PolytopeSamplers.grad_potential!(gU, c, z)
        dKdp = zeros(k)
        dKdz = zeros(k)
        @test PolytopeSamplers.grad_kinetic!(dKdp, dKdz, c, z, p)
        fdU = [(U(z + ε * e(i)) - U(z - ε * e(i))) / 2ε for i in 1:k]
        fdKz = [(K(z + ε * e(i), p) - K(z - ε * e(i), p)) / 2ε for i in 1:k]
        fdKp = [(K(z, p + ε * e(i)) - K(z, p - ε * e(i))) / 2ε for i in 1:k]
        @test norm(gU - fdU) < 1.0e-6 * norm(fdU)
        @test norm(dKdz - fdKz) < 1.0e-6 * norm(fdKz)
        @test norm(dKdp - fdKp) < 1.0e-6 * norm(fdKp)
        # Outside the polytope there is no metric and the Hamiltonian is infinite
        @test PolytopeSamplers.hamiltonian(fresh(), fill(100.0, k), p) == Inf
    end

    @testset "step size schedule" begin
        c = PolytopeSamplers.chain_state(CRHMC(), w)
        @test c.h == 1.0e-3
        rng = Xoshiro(1)
        for _ in 1:3_000
            PolytopeSamplers.step!(rng, c)
        end
        # The step size grows to the initial step size and then only shrinks
        @test c.warmed_up
        @test 1.0e-3 <= c.h <= 0.2
        @test c.accepted / c.proposed > 0.8
        # A larger initial step size is shrunk back by rejections
        c = PolytopeSamplers.chain_state(CRHMC(initial_step_size = 5.0), w)
        for _ in 1:3_000
            PolytopeSamplers.step!(rng, c)
        end
        @test c.h < 5.0
    end

    @testset "the implicit midpoint rule rarely fails" begin
        # A step whose integration fails is rejected; failures near the boundary bias the
        # chain toward the interior (2% of steps failing gave standard deviations 4 Monte
        # Carlo standard errors low), so they must stay rare
        wc = warmup(Polytope(fill(-1.0, 10), fill(1.0, 10)))
        c = PolytopeSamplers.chain_state(CRHMC(), wc)
        rng = Xoshiro(1)
        for _ in 1:2_000
            PolytopeSamplers.step!(rng, c)
        end
        failures = 0
        z = similar(c.z)
        p = similar(c.p)
        for _ in 1:4_000
            copyto!(z, c.z)
            copyto!(p, c.p)
            PolytopeSamplers.refresh_momentum!(rng, c, c.momentum)
            failures += PolytopeSamplers.implicit_midpoint!(c, c.h) == 0
            copyto!(c.z, z)
            copyto!(c.p, p)
            PolytopeSamplers.step!(rng, c)
        end
        @test failures / 4_000 < 0.005
    end

    @testset "options" begin
        @test_throws ArgumentError CRHMC(initial_step_size = 0)
        @test_throws ArgumentError CRHMC(effective_step_size = -1)
        @test_throws ArgumentError CRHMC(max_ode_steps = 0)
        @test_throws ArgumentError CRHMC(implicit_tol = 0)
        @test sprint(show, CRHMC()) == "CRHMC()"
    end
end
