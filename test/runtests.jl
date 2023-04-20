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

function get_pkg_version(name::AbstractString)
    deps = Pkg.dependencies()
    p = only(x for x in values(deps) if x.name == name)
    return p.version
end

## Test sets

@testset verbose = true "ImplicitDifferentiation.jl" begin
    @testset verbose = true "Code quality (Aqua.jl)" begin
        Aqua.test_all(ImplicitDifferentiation)
    end
    @testset verbose = true "Code formatting (JuliaFormatter.jl)" begin
        @test format(ImplicitDifferentiation; verbose=true, overwrite=false)
    end
    @testset verbose = true "Code correctness (JET.jl)" begin
        if get_pkg_version("JET") >= v"0.7.11"
            JET.test_package("InferOpt"; toplevel_logger=nothing)
        else
            @test string(JET.report_package(InferOpt)) == "No errors detected\n"
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
