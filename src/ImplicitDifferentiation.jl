module ImplicitDifferentiation

using Krylov: KrylovStats, gmres
using LinearOperators: LinearOperators, LinearOperator
using LinearAlgebra: lu, SingularException, issuccess
using PrecompileTools: @compile_workload
using Requires: @require
using SimpleUnPack: @unpack

include("utils.jl")
include("forward.jl")
include("conditions.jl")
include("linear_solver.jl")
include("implicit_function.jl")

export ImplicitFunction
export IterativeLinearSolver, DirectLinearSolver
export HandleByproduct, ReturnByproduct

@static if !isdefined(Base, :get_extension)
    include("../ext/ImplicitDifferentiationChainRulesExt.jl")
    function __init__()
        @require ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210" begin
            include("../ext/ImplicitDifferentiationForwardDiffExt.jl")
        end
        @require ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267" begin
            include("../ext/ImplicitDifferentiationReverseDiffExt.jl")
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
