## Imports

using Aqua
using Documenter
using ForwardDiff: ForwardDiff
using ImplicitDifferentiation
using JET
using JuliaFormatter
using Random
using Test
using Zygote: Zygote

DocMeta.setdocmeta!(
    ImplicitDifferentiation, :DocTestSetup, :(using ImplicitDifferentiation); recursive=true
)

EXAMPLES_DIR_JL = joinpath(dirname(@__DIR__), "examples")

## Test sets

@testset verbose = true "ImplicitDifferentiation.jl" begin
    @testset verbose = false "Code quality (Aqua.jl)" begin
        Aqua.test_all(
            ImplicitDifferentiation; ambiguities=false, deps_compat=(check_extras = false)
        )
    end
    @testset verbose = true "Formatting (JuliaFormatter.jl)" begin
        @test format(ImplicitDifferentiation; verbose=false, overwrite=false)
    end
    @testset verbose = true "Static checking (JET.jl)" begin
        JET.test_package(ImplicitDifferentiation; target_defined_modules=true)
    end
    @testset verbose = false "Doctests (Documenter.jl)" begin
        doctest(ImplicitDifferentiation)
    end
    @testset verbose = true "Examples" begin
        @info "Example tests"
        for file in readdir(EXAMPLES_DIR_JL)
            @info "$file"
            @testset "$file" begin
                include(joinpath(EXAMPLES_DIR_JL, file))
            end
        end
    end
    @testset verbose = true "Systematic" begin
        @info "Systematic tests"
        include("systematic.jl")
    end
end
