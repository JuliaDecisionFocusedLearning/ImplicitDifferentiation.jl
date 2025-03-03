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

Documenter.DocMeta.setdocmeta!(
    ImplicitDifferentiation, :DocTestSetup, :(using ImplicitDifferentiation); recursive=true
)

EXAMPLES_DIR_JL = joinpath(dirname(@__DIR__), "examples")

## Test sets

@testset verbose = true "ImplicitDifferentiation.jl" begin
    @testset "Code quality (Aqua.jl)" begin
        Aqua.test_all(
            ImplicitDifferentiation; ambiguities=false, deps_compat=(check_extras = false)
        )
    end
    @testset "Formatting (JuliaFormatter.jl)" begin
        @test format(ImplicitDifferentiation; verbose=false, overwrite=false)
    end
    @testset "Static checking (JET.jl)" begin
        JET.test_package(ImplicitDifferentiation; target_defined_modules=true)
    end
    @testset "Imports (ExplicitImports.jl)" begin
        @test check_no_implicit_imports(ImplicitDifferentiation) === nothing
        @test check_no_stale_explicit_imports(ImplicitDifferentiation) === nothing
        @test check_all_explicit_imports_via_owners(ImplicitDifferentiation) === nothing
        @test_broken check_all_explicit_imports_are_public(ImplicitDifferentiation) ===
            nothing
        @test check_all_qualified_accesses_via_owners(ImplicitDifferentiation) === nothing
        @test check_all_qualified_accesses_are_public(ImplicitDifferentiation) === nothing
        @test check_no_self_qualified_accesses(ImplicitDifferentiation) === nothing
    end
    @testset "Doctests (Documenter.jl)" begin
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
end;
