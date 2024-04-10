"""
    ImplicitDifferentiation

A Julia package for automatic differentiation of implicit functions.

Its main export is the type [`ImplicitFunction`](@ref).
"""
module ImplicitDifferentiation

using ADTypes: AbstractADType
using DifferentiationInterface:
    jacobian,
    prepare_pushforward,
    prepare_pullback,
    pushforward!!,
    value_and_pullback!!_split
using Krylov: block_gmres, gmres
using LinearOperators: LinearOperator
using LinearAlgebra: factorize, lu

include("implicit_function.jl")
include("operators.jl")

export ImplicitFunction

end
