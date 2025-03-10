## Linear solver

"""
    KrylovLinearSolver

Callable object that can solve linear systems `Ax = b` and `AX = B` in the same way as the built-in `\\`.
Uses an iterative solver from [Krylov.jl](https://github.com/JuliaSmoothOptimizers/Krylov.jl) under the hood.

# Constructor

    KrylovLinearSolver(; verbose=true)

If `verbose` is `true`, the solver logs a warning in case of failure.
Otherwise it will fail silently, and may return solutions that do not exactly satisfy the linear system.

# Callable behavior

    (::KylovLinearSolver)(A, b::AbstractVector)

Solve a linear system with a single right-hand side.

    (::KrylovLinearSolver)(A, B::AbstractMatrix)

Solve a linear system with multiple right-hand sides.
"""
Base.@kwdef struct KrylovLinearSolver
    verbose::Bool = true
end

function (solver::KrylovLinearSolver)(A, b::AbstractVector)
    x, stats = gmres(A, b)
    if !stats.solved || stats.inconsistent
        solver.verbose &&
            @warn "Failed to solve the linear system in the implicit function theorem with `Krylov.gmres`" stats
    end
    return x
end

function (solver::KrylovLinearSolver)(A, B::AbstractMatrix)
    # X, stats = block_gmres(A, B)  # https://github.com/JuliaSmoothOptimizers/Krylov.jl/issues/854
    X = mapreduce(hcat, eachcol(B)) do b
        solver(A, b)
    end
    return X
end

## Representation

abstract type AbstractRepresentation end

"""
    MatrixRepresentation

Specify that the matrices involved in the implicit function theorem should be represented explicitly, with all their coefficients.

# See also

- [`ImplicitFunction`](@ref)
- [`OperatorRepresentation`](@ref)
"""
struct MatrixRepresentation <: AbstractRepresentation end

"""
    OperatorRepresentation

Specify that the matrices involved in the implicit function theorem should be represented lazily, as linear operators from [LinearOperators.jl](https://github.com/JuliaSmoothOptimizers/LinearOperators.jl).

# See also

- [`ImplicitFunction`](@ref)
- [`MatrixRepresentation`](@ref)
"""
struct OperatorRepresentation <: AbstractRepresentation end

## Preparation

abstract type AbstractPreparation end

"""
    ForwardPreparation

Specify that the derivatives of the conditions should be prepared for subsequent forward-mode differentiation of the implicit function.
"""
struct ForwardPreparation <: AbstractPreparation end

"""
    ReversePreparation

Specify that the derivatives of the conditions should be prepared for subsequent reverse-mode differentiation of the implicit function.
"""
struct ReversePreparation <: AbstractPreparation end

"""
    BothPreparation

Specify that the derivatives of the conditions should be prepared for subsequent forward- or reverse-mode differentiation of the implicit function.
"""
struct BothPreparation <: AbstractPreparation end

"""
    NoPreparation

Specify that the derivatives of the conditions should not be prepared for subsequent differentiation of the implicit function.
"""
struct NoPreparation <: AbstractPreparation end

function chainrules_suggested_backend end
