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
using Krylov: Krylov, krylov_workspace, krylov_solve!, solution
using LinearOperators: LinearOperator
using LinearMaps: FunctionMap
using LinearAlgebra: factorize

include("utils.jl")
include("settings.jl")
include("implicit_function.jl")
include("preparation.jl")
include("execution.jl")
include("callable.jl")

export MatrixRepresentation, OperatorRepresentation
export IterativeLinearSolver, DirectLinearSolver
export ImplicitFunction
export prepare_implicit

end
