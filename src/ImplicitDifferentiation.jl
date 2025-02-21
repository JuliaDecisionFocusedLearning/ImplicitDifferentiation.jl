"""
    ImplicitDifferentiation

A Julia package for automatic differentiation of implicit functions.

Its main export is the type [`ImplicitFunction`](@ref).
"""
module ImplicitDifferentiation

using ADTypes: AbstractADType
using DifferentiationInterface:
    Constant,
    jacobian,
    prepare_jacobian,
    prepare_pullback,
    prepare_pullback_same_point,
    prepare_pushforward,
    prepare_pushforward_same_point,
    pullback!,
    pushforward!,
    unwrap
using Krylov: block_gmres, gmres
using LinearOperators: LinearOperator
using LinearAlgebra: axpby!, factorize, lu

include("preparation.jl")
include("linear_solver.jl")
include("implicit_function.jl")
include("execution.jl")

export ImplicitFunction, KrylovLinearSolver

end
