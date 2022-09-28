using Test, QMC
using Statistics
using Aqua
using Measurements
using RandomNumbers
import Distributions: Dirichlet
# Aqua.test_all(QMC, ambiguities=false, stale_deps=false)

THRESHOLD = 1.282  # 80% Two-sided CI of the t-distribution with infinite dofs
Z_THRESHOLD = 1.0  # 68% CI of standard normal

@testset "Probability Vectors" begin
    @testset "Small Vector of size $n; seed $seed" for n in [5, 7, 9, 10, 15, 25, 50, 100, 250, 1000], seed in [1234, 4321, 5555]
        rng = Xorshifts.Xoroshiro128Plus(seed)
        prior = Dirichlet(n, 1.0)
        p = rand(rng, prior)
        mean_ = p' * collect(1:size(p,1)) / sum(p)
        var_ = (p' * (collect(1:size(p,1)) .^ 2) / sum(p)) - mean_^2

        N = 1_000_000

        @testset "ProbabilityVector" begin
            pvec = ProbabilityVector(p)
            X = [rand(rng, pvec) for _ in 1:N]

            @test isapprox(mean_, mean(X), rtol=0.05)
            @test isapprox(var_, var(X), rtol=0.05)

            @test abs((mean(X) - mean_) / sqrt(var_)) < Z_THRESHOLD
        end

        @testset "ProbabilityHeap" begin
            pvec = ProbabilityHeap(p)
            X = [rand(rng, pvec) for _ in 1:N]

            @test isapprox(mean_, mean(X), rtol=0.05)
            @test isapprox(var_, var(X), rtol=0.05)

            @test abs((mean(X) - mean_) / sqrt(var_)) < Z_THRESHOLD
        end


        @testset "ProbabilityAlias" begin
            pvec = ProbabilityAlias(float.(p))
            X = [rand(rng, pvec) for _ in 1:N]

            @test isapprox(mean_, mean(X), rtol=0.05)
            @test isapprox(var_, var(X), rtol=0.05)

            @test abs((mean(X) - mean_) / sqrt(var_)) < Z_THRESHOLD
        end
    end
end

H = @inferred LTFIM((10,), 1.0, 1.0, 1.0)
@inferred BinaryGroundState BinaryGroundState(H, 1000)
@inferred BinaryThermalState BinaryThermalState(H, 1000)

# include("ltfim.jl")
include("tfim.jl")
