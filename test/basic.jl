# Basic use

using Enzyme
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
    a[1] = first(x)
    return x
end;

x = rand(2, 3)

@testset verbose = true "Automatic diff fails" begin
    @test myidentity(x) ≈ x
    @test_throws MethodError ForwardDiff.jacobian(myidentity, rand(2))
    @test_throws ErrorException Zygote.jacobian(myidentity, rand(2))
end

forward(x) = myidentity(x), 0
conditions(x, y, z) = y .- x;
implicit = ImplicitFunction(forward, conditions);

@testset verbose = true "Implicit diff works" begin
    implicit(x) ≈ x
    ForwardDiff.jacobian(implicit, x) ≈ I
    Zygote.jacobian(implicit, x)[1] ≈ I
end
