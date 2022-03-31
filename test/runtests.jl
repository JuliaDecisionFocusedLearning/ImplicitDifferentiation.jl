## Imports

using Aqua
using ImplicitDifferentiation
using Random
using Test

Random.seed!(63)

## Test sets

@testset verbose = true "ImplicitDifferentiation.jl" begin
    @testset verbose = true "Unconstrained optimization" begin
        include("1_unconstrained_optimization.jl")
    end

    @testset verbose = true "Constrained optimization" begin
        include("2_constrained_optimization.jl")
    end

    @testset verbose = true "Code quality" begin
        Aqua.test_all(ImplicitDifferentiation)
    end
end
