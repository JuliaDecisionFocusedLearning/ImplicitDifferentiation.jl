## Imports

using Aqua
using ImplicitDifferentiation
using Test

## Test sets

@testset verbose = true "ImplicitDifferentiation.jl" begin
    @testset verbose = true "Unconstrained optimization" begin
        include("1_unconstrained_optimization.jl")
    end
    @testset verbose = true "Code quality" begin
        Aqua.test_all(ImplicitDifferentiation)
    end
end
