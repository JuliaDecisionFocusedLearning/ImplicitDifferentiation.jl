using ChainRulesCore
using ChainRulesTestUtils
using ForwardDiff
using ImplicitDifferentiation
using JET
using LinearAlgebra
using Random
using Test
using Zygote

Random.seed!(63);

function mysqrt(x::AbstractArray)
    a = [0.0]
    a[1] = first(x)
    return sqrt.(x)
end

forward(x) = mysqrt(x), 0
conditions(x, y, z) = y .^ 2 .- x
implicit = ImplicitFunction(forward, conditions)

# Skipped because of https://github.com/JuliaDiff/ChainRulesTestUtils.jl/issues/232 and because it detects weird type instabilities
@testset verbose = true "ChainRulesTestUtils.jl" begin
    @test_skip test_rrule(implicit, x)
    @test_skip test_rrule(implicit, X)
end

@testset verbose = true "Vectors" begin
    x = rand(2)
    y, _ = implicit(x)
    J = Diagonal(0.5 ./ sqrt.(x))

    @testset "Exactness" begin
        @test (first ∘ implicit)(x) ≈ sqrt.(x)
        @test ForwardDiff.jacobian(first ∘ implicit, x) ≈ J
        @test Zygote.jacobian(first ∘ implicit, x)[1] ≈ J
    end

    @testset verbose = true "Forward inference" begin
        x_and_dx = ForwardDiff.Dual.(x, ((0, 0),))
        @test (@inferred implicit(x_and_dx)) == implicit(x_and_dx)
        y_and_dy, _ = implicit(x_and_dx)
        @test size(y_and_dy) == size(y)
    end
    @testset "Reverse type inference" begin
        _, pullback = @inferred rrule(Zygote.ZygoteRuleConfig(), implicit, x)
        dy, dz = zero(implicit(x)[1]), 0
        @test (@inferred pullback((dy, dz))) == pullback((dy, dz))
        _, dx = pullback((dy, dz))
        @test size(dx) == size(x)
    end
end

@testset verbose = true "Arrays" begin
    X = rand(2, 3, 4)
    Y, _ = implicit(X)
    JJ = Diagonal(0.5 ./ sqrt.(vec(X)))

    @testset "Exactness" begin
        @test (first ∘ implicit)(X) ≈ sqrt.(X)
        @test ForwardDiff.jacobian(first ∘ implicit, X) ≈ JJ
        @test Zygote.jacobian(first ∘ implicit, X)[1] ≈ JJ
    end

    @testset "Forward type inference" begin
        X_and_dX = ForwardDiff.Dual.(X, ((0, 0),))
        @test (@inferred implicit(X_and_dX)) == implicit(X_and_dX)
        Y_and_dY, _ = implicit(X_and_dX)
        @test size(Y_and_dY) == size(Y)
    end

    @testset "Reverse type inference" begin
        _, pullback = @inferred rrule(Zygote.ZygoteRuleConfig(), implicit, X)
        dY, dZ = zero(implicit(X)[1]), 0
        @test (@inferred pullback((dY, dZ))) == pullback((dY, dZ))
        _, dX = pullback((dY, dZ))
        @test size(dX) == size(X)
    end
end
