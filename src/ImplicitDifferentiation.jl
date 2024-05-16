"""
    ImplicitDifferentiation

A Julia package for automatic differentiation of implicit functions.

Its main export is the type [`ImplicitFunction`](@ref).
"""
module ImplicitDifferentiation

using ADTypes: AbstractADType
using DifferentiationInterface:
    jacobian,
    prepare_pushforward_same_point,
    prepare_pullback_same_point,
    pullback!,
    pushforward!
using Krylov: block_gmres, gmres
using LinearOperators: LinearOperator
using LinearAlgebra: factorize, lu

include("implicit_function.jl")
include("operators.jl")

export ImplicitFunction

end
