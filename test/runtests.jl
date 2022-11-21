## Imports

using Aqua
using ImplicitDifferentiation
using Random
using Test

## Test sets

@testset verbose = true "ImplicitDifferentiation.jl" begin
    @testset verbose = true "Code quality (Aqua)" begin
        Aqua.test_all(ImplicitDifferentiation)
    end
    @testset verbose = true "Unconstrained optimization" begin
        include("1_unconstrained_optimization.jl")
    end
    @testset verbose = true "Sparse linear regression" begin
        include("2_sparse_linear_regression.jl")
    end
    @testset verbose = true "Optimal transport" begin
        include("3_optimal_transport.jl")
    end
end
