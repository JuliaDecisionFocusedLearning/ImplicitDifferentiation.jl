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

    @testset "Call" begin
        @test (@inferred implicit(x)) ≈ sqrt.(x)
        if VERSION >= v"1.7"
            test_opt(implicit, (typeof(x),))
        end
    end

    @testset verbose = true "Forward" begin
        @test ForwardDiff.jacobian(implicit, x) ≈ J
        x_and_dx = ForwardDiff.Dual.(x, ((0, 0),))
        @test (@inferred implicit(x_and_dx)) == implicit(x_and_dx)
        y_and_dy, _ = implicit(x_and_dx)
        @test size(y_and_dy) == size(y)
    end

    @testset "Reverse" begin
        @test Zygote.jacobian(implicit, x)[1] ≈ J
        for return_byproduct in (true, false)
            _, pullback = @inferred rrule(
                Zygote.ZygoteRuleConfig(), implicit, x, Val(return_byproduct)
            )
            dy, dz = zero(implicit(x)), 0
            if return_byproduct
                @test (@inferred pullback((dy, dz))) == pullback((dy, dz))
                _, dx = pullback((dy, dz))
                @test size(dx) == size(x)
            else
                @test (@inferred pullback(dy)) == pullback(dy)
                _, dx = pullback(dy)
                @test size(dx) == size(x)
            end
        end
    end
end

@testset verbose = true "Arrays" begin
    X = rand(2, 3, 4)
    Y, _ = implicit(X)
    JJ = Diagonal(0.5 ./ sqrt.(vec(X)))

    @testset "Call" begin
        @test (@inferred implicit(X)) ≈ sqrt.(X)
        if VERSION >= v"1.7"
            test_opt(implicit, (typeof(X),))
        end
    end

    @testset "Forward" begin
        @test ForwardDiff.jacobian(implicit, X) ≈ JJ
        X_and_dX = ForwardDiff.Dual.(X, ((0, 0),))
        @test (@inferred implicit(X_and_dX)) == implicit(X_and_dX)
        Y_and_dY, _ = implicit(X_and_dX)
        @test size(Y_and_dY) == size(Y)
    end

    @testset "Reverse" begin
        @test Zygote.jacobian(implicit, X)[1] ≈ JJ
        for return_byproduct in (true, false)
            _, pullback = @inferred rrule(
                Zygote.ZygoteRuleConfig(), implicit, X, Val(return_byproduct)
            )
            dY, dZ = zero(implicit(X)), 0
            if return_byproduct
                @test (@inferred pullback((dY, dZ))) == pullback((dY, dZ))
                _, dX = pullback((dY, dZ))
                @test size(dX) == size(X)
            else
                @test (@inferred pullback(dY)) == pullback(dY)
                _, dX = pullback(dY)
                @test size(dX) == size(X)
            end
        end
    end
end
