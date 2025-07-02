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
struct IterativeLinearSolver{K}
    kwargs::K
    function IterativeLinearSolver(; kwargs...)
        return new{typeof(kwargs)}(kwargs)
    end
end

function (solver::IterativeLinearSolver)(A, b)
    sol, info = linsolve(A, b; solver.kwargs...)
    @assert info.converged == 1
    return sol
end

function Base.show(io::IO, linear_solver::IterativeLinearSolver)
    (; kwargs) = linear_solver
    print(io, repr(IterativeLinearSolver; context=io), "(;")
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

function chainrules_suggested_backend end
