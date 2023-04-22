## Imports

using Aqua
using Documenter
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

function markdown_title(path)
    title = "?"
    open(path, "r") do file
        for line in eachline(file)
            if startswith(line, '#')
                title = strip(line, [' ', '#'])
                break
            end
        end
    end
    return title
end

EXAMPLES_DIR_JL = joinpath(dirname(@__DIR__), "examples")

## Test sets

@testset verbose = true "ImplicitDifferentiation.jl" begin
    @testset verbose = true "Code quality (Aqua.jl)" begin
        Aqua.test_all(ImplicitDifferentiation)
    end
    @testset verbose = true "Formatting (JuliaFormatter.jl)" begin
        @test format(ImplicitDifferentiation; verbose=true, overwrite=false)
    end
    @testset verbose = true "Static checking (JET.jl)" begin
        if VERSION >= v"1.8"
            JET.test_package(ImplicitDifferentiation; toplevel_logger=nothing)
        end
    end
    @testset verbose = true "Doctests (Documenter.jl)" begin
        doctest(ImplicitDifferentiation)
    end
    @testset verbose = true "Basic" begin
        include("basic.jl")
    end
    # TODO: toggle
    # for file in readdir(EXAMPLES_DIR_JL)
    #     path = joinpath(EXAMPLES_DIR_JL, file)
    #     title = markdown_title(path)
    #     @testset verbose = true "$title" begin
    #         include(path)
    #     end
    # end
end
