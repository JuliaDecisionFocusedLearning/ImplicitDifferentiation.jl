## Imports

using Aqua
using ImplicitDifferentiation
using Random
using Test

## Test sets

@testset verbose = true "ImplicitDifferentiation.jl" begin
    @testset verbose = true "Unconstrained optimization" begin
        include("1_unconstrained_optimization.jl")
    end
    @testset verbose = true "Constrained optimization" begin
        include("2_constrained_optimization.jl")
    end
    @testset verbose = true "Entropy regularized optimal transport" begin
        include("3_optimal_transport.jl")
    end
    @testset verbose = true "Custom structs" begin
        include("4_struct.jl")
    end
    @testset verbose = true "Code quality" begin
        Aqua.test_all(ImplicitDifferentiation)
    end
end
