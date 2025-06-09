"""
    ImplicitDifferentiation

A Julia package for automatic differentiation of implicit functions.

Its main export is the type [`ImplicitFunction`](@ref).
"""
module ImplicitDifferentiation

using ADTypes:
    ADTypes, AbstractADType, AbstractMode, ForwardMode, ReverseMode, ForwardOrReverseMode
using DifferentiationInterface:
    Constant,
    jacobian,
    prepare_jacobian,
    prepare_pullback,
    prepare_pullback_same_point,
    prepare_pushforward,
    prepare_pushforward_same_point,
    pullback!,
    pullback,
    pushforward!,
    pushforward
using Krylov: gmres
using LinearOperators: LinearOperator
using LinearMaps: FunctionMap
using LinearAlgebra: factorize

include("utils.jl")
include("settings.jl")
include("preparation.jl")
include("implicit_function.jl")
include("execution.jl")

export IterativeLinearSolver
export MatrixRepresentation, OperatorRepresentation
export NoPreparation, ForwardPreparation, ReversePreparation, BothPreparation
export ImplicitFunction

end
