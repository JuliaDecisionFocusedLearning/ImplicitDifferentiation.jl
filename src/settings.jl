## Linear solver

"""
    AbstractSolver

Abstract supertype for linear solvers to be used within an [`ImplicitFunction`](@ref).

# Interface

The only requirement on any concrete subtype `MySolver` is that it should be callable, with the following signature:

    (s::MySolver)(A, Aᵀ, x0, b)

This method should solve the linear system `Ax = b`, possibly using the transpose `Aᵀ` or starting from an initial point `x0`.
Depending on the [`AbstractRepresentation`](@ref) chosen for the `ImplicitFunction`, the arguments `A` and `Aᵀ` might be either matrix-like or functional objects.

# Implementations

This package provides three default implementations:

- [`DirectLinearSolver`](@ref)
- [`IterativeLinearSolver`](@ref)
- [`IterativeLeastSquaresSolver`](@ref)
"""
abstract type AbstractSolver end

"""
    DirectLinearSolver <: AbstractSolver

Specify that linear systems `Ax = b` should be solved with a direct method.

!!! warning
    Can only be used when the `solver` and the `conditions` both output an `AbstractVector`.

    Additionally, this solver requires a [`MatrixRepresentation`](@ref) of the matrix `A`. To do so,
    use the `representation` keyword of the [`ImplicitFunction`](@ref) constructor :

        f = ImplicitFunction(
            forward,
            conditions;
            solver = DirectLinearSolver(),
            representation = MatrixRepresentation()
        )

# See also

- [`ImplicitFunction`](@ref)
- [`AbstractSolver`](@ref)
"""
struct DirectLinearSolver <: AbstractSolver end

function (solver::DirectLinearSolver)(
    A::Union{AbstractMatrix,Factorization,Number},
    _Aᵀ,
    b::AbstractVector,
    x0::AbstractVector,
)
    return A \ b
end

abstract type AbstractIterativeSolver <: AbstractSolver end

"""
    IterativeLinearSolver <: AbstractSolver

Specify that linear systems `Ax = b` should be solved with an iterative method.

!!! warning
    Can only be used when the `solver` and the `conditions` both output `AbstractArray`s with the same type and length.

# See also

- [`ImplicitFunction`](@ref)
- [`AbstractSolver`](@ref)
"""
struct IterativeLinearSolver{K} <: AbstractIterativeSolver
    kwargs::K
    function IterativeLinearSolver(; kwargs...)
        return new{typeof(kwargs)}(kwargs)
    end
end

function (solver::IterativeLinearSolver)(A, _Aᵀ, b, x0)
    sol, info = linsolve(A, b, x0; solver.kwargs...)
    return sol
end

"""
    IterativeLeastSquaresSolver <: AbstractSolver

Specify that linear systems `Ax = b` should be solved with an iterative least-squares method.

!!! tip
    Can be used when the `solver` and the `conditions` output `AbstractArray`s with different types or different lengths.

!!! warning
    To ensure performance, remember to specify both `backends` used to differentiate `condtions`.

# See also

- [`ImplicitFunction`](@ref)
- [`AbstractSolver`](@ref)
"""
struct IterativeLeastSquaresSolver{K} <: AbstractIterativeSolver
    kwargs::K
    function IterativeLeastSquaresSolver(; kwargs...)
        return new{typeof(kwargs)}(kwargs)
    end
end

function (solver::IterativeLeastSquaresSolver)(A, Aᵀ, b, x0)
    sol, info = lssolve((A, Aᵀ), b; solver.kwargs...)
    return sol
end

function Base.show(io::IO, linear_solver::AbstractIterativeSolver)
    (; kwargs) = linear_solver
    T = if linear_solver isa IterativeLinearSolver
        IterativeLinearSolver
    else
        IterativeLeastSquaresSolver
    end
    print(io, repr(T; context=io), "(;")
    for p in pairs(kwargs)
        print(io, " ", p[1], "=", repr(p[2]; context=io), ",")
    end
    return print(io, ")")
end

## Representation

abstract type AbstractRepresentation end

"""
    MatrixRepresentation

Specify that the matrix `A` involved in the implicit function theorem should be represented explicitly, with all its coefficients.

# See also

- [`ImplicitFunction`](@ref)
- [`OperatorRepresentation`](@ref)
"""
struct MatrixRepresentation <: AbstractRepresentation end

"""
    OperatorRepresentation

Specify that the matrix `A` involved in the implicit function theorem should be represented lazily, as a function.

# See also

- [`ImplicitFunction`](@ref)
- [`MatrixRepresentation`](@ref)
"""
struct OperatorRepresentation <: AbstractRepresentation end

## Backends

function suggested_forward_backend end
function suggested_reverse_backend end
