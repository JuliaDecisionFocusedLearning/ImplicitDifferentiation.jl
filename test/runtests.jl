## Imports

using Aqua
using GalacticOptim
using ImplicitDifferentiation
using IterativeSolvers
using Optim
using Test
using Zygote

using Statistics

## Test sets

@testset verbose = true "ImplicitDifferentiation.jl" begin
    @testset verbose = true "ImplicitFunction" begin
        include("implicit_function.jl")
    end
    @testset verbose = true "Code quality" begin
        Aqua.test_all(ImplicitDifferentiation)
    end
end
