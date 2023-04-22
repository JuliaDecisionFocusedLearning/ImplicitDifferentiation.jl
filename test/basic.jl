# Basic use

using ForwardDiff
using ImplicitDifferentiation
using JET
using LinearAlgebra
using Random
using Test
using Zygote

Random.seed!(63);

function myidentity(x)
    a = [0.0]
    a[1] = 1.0
    return Float64.(x)
end;

x = rand(2, 3)

@testset verbose = true "Automatic diff fails" begin
    @test myidentity(x) ≈ x
    @test_throws MethodError ForwardDiff.jacobian(myidentity, rand(2))
    @test_throws ErrorException Zygote.jacobian(myidentity, rand(2))
end

forward(x) = myidentity(x), nothing
conditions(x, y, z) = y - x;
implicit = ImplicitFunction(forward, conditions);

@testset verbose = true "Implicit diff succeeds" begin
    @test implicit(x) ≈ x
    @test ForwardDiff.jacobian(implicit, x) ≈ I
    @test Zygote.jacobian(implicit, x)[1] ≈ I
end

@report_call implicit(x)
@report_call ForwardDiff.jacobian(implicit, x)
@report_call Zygote.jacobian(implicit, x)

@testset verbose = true "Correctness and type-stability" begin
    @test_call implicit(x)
    # @test_call ForwardDiff.jacobian(implicit, x)
    # @test_call Zygote.jacobian(implicit, x)
end
