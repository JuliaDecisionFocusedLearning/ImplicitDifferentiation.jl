## Imports

using Aqua
using ImplicitDifferentiation
using JET
using Random
using Test

## Test sets

@testset verbose = true "ImplicitDifferentiation.jl" begin
    @testset verbose = true "Code quality" begin
        @testset verbose = true "JET" begin
            jet_report = JET.report_package(ImplicitDifferentiation)
            @test string(jet_report) == "No errors detected\n"
        end
        @testset verbose = true "Aqua" begin
            Aqua.test_all(ImplicitDifferentiation)
        end
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
