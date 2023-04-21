## Imports

using Aqua
using Documenter
using ForwardDiffChainRules
using ImplicitDifferentiation
using JET
using JuliaFormatter
using Pkg
using Random
using Test
using Zygote

DocMeta.setdocmeta!(
    ImplicitDifferentiation, :DocTestSetup, :(using ImplicitDifferentiation); recursive=true
)

## Test sets

@testset verbose = true "ImplicitDifferentiation.jl" begin
    @testset verbose = true "Code quality (Aqua.jl)" begin
        Aqua.test_all(ImplicitDifferentiation)
    end
    @testset verbose = true "Code formatting (JuliaFormatter.jl)" begin
        @test format(ImplicitDifferentiation; verbose=true, overwrite=false)
    end
    @testset verbose = true "Code correctness (JET.jl)" begin
        if VERSION >= v"1.8"
            @test_broken JET.test_package(ImplicitDifferentiation; toplevel_logger=nothing)
        end
    end
    @testset verbose = true "Doctests (Documenter.jl)" begin
        doctest(ImplicitDifferentiation)
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
