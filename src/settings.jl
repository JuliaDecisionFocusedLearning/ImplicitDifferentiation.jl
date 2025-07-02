## Linear solver

"""
    DirectLinearSolver

Specify that linear systems `Ax = b` should be solved with a direct method.

# See also

- [`ImplicitFunction`](@ref)
- [`IterativeLinearSolver`](@ref)
"""
struct DirectLinearSolver end

function (solver::DirectLinearSolver)(A, b::AbstractVector)
    return A \ b
end

"""
    IterativeLinearSolver

Specify that linear systems `Ax = b` should be solved with an iterative method.

# See also

- [`ImplicitFunction`](@ref)
- [`DirectLinearSolver`](@ref)
"""
struct IterativeLinearSolver{A,K}
    algorithm::A
    kwargs::K
    function IterativeLinearSolver(algorithm=GMRES(); kwargs...)
        return new{typeof(algorithm),typeof(kwargs)}(algorithm, kwargs)
    end
end

function (solver::IterativeLinearSolver)(A, b)
    x0 = zero(b)
    sol, info = linsolve(A, b, x0, solver.algorithm; solver.kwargs...)
    @assert info.converged == 1
    return sol
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

function chainrules_suggested_backend end
