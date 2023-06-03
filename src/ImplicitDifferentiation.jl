module ImplicitDifferentiation

using AbstractDifferentiation:
    LazyJacobian, ReverseRuleConfigBackend, lazy_jacobian, pushforward_function
using Krylov: KrylovStats, gmres
using LinearOperators: LinearOperators, LinearOperator
using LinearAlgebra: lu
using Requires: @require
using SimpleUnPack: @unpack

include("utils.jl")
include("implicit_function.jl")

export ImplicitFunction

@static if !isdefined(Base, :get_extension)
    include("../ext/ImplicitDifferentiationChainRulesExt.jl")
    function __init__()
        @require ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210" begin
            include("../ext/ImplicitDifferentiationForwardDiffExt.jl")
        end
        @require StaticArrays = "90137ffa-7385-5640-81b9-e52037218182" begin
            include("../ext/ImplicitDifferentiationStaticArraysExt.jl")
        end
    end
end

end
