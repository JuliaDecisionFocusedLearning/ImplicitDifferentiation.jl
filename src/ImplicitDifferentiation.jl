"""
    ImplicitDifferentiation

A Julia package for automatic differentiation of implicit functions.

Its main export is the type [`ImplicitFunction`](@ref).
"""
module ImplicitDifferentiation

using AbstractDifferentiation: AbstractBackend, pushforward_function, pullback_function
using Krylov: gmres
using LinearOperators: LinearOperators, LinearOperator
using LinearAlgebra: issuccess, lu
using PrecompileTools: @compile_workload
using Requires: @require
using SimpleUnPack: @unpack

include("utils.jl")
include("linear_solver.jl")
include("implicit_function.jl")
include("operators.jl")

export ImplicitFunction
export AbstractLinearSolver, IterativeLinearSolver, DirectLinearSolver

@static if !isdefined(Base, :get_extension)
    # Loaded unconditionally on Julia < 1.9
    include("../ext/ImplicitDifferentiationChainRulesCoreExt.jl")
    function __init__()
        # Loaded conditionally on Julia < 1.9
        @require ComponentArrays = "b0b7db55-cfe3-40fc-9ded-d10e2dbeff66" begin
            include("../ext/ImplicitDifferentiationComponentArraysExt.jl")
        end
        @require ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210" begin
            include("../ext/ImplicitDifferentiationForwardDiffExt.jl")
        end
        @require StaticArrays = "90137ffa-7385-5640-81b9-e52037218182" begin
            include("../ext/ImplicitDifferentiationStaticArraysExt.jl")
        end
        @require Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f" begin
            include("../ext/ImplicitDifferentiationZygoteExt.jl")
        end
    end
end

end
