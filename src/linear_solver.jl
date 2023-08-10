"""
    AbstractLinearSolver

All linear solvers used within an `ImplicitFunction` must satisfy this interface.

It can be useful to roll out your own solver if you need more fine-grained control on convergence / speed / behavior in case of singularity.
Check out the source code of `IterativeLinearSolver` and `DirectLinearSolver` for implementation examples. 

# Required methods

- `presolve(linear_solver, A, y)`: Returns a matrix-like object `A` for which it is cheaper to solve several linear systems with different vectors `b` of type similar to `y` (a typical example would be to perform LU factorization).
- `solve(linear_solver, A, b)`: Returns a vector `x` satisfying `Ax = b`. If the linear system has not been solved to satisfaction, every element of `x` should be a `NaN` of the appropriate floating point type.
"""
abstract type AbstractLinearSolver end

"""
    IterativeLinearSolver

An implementation of `AbstractLinearSolver` using `Krylov.gmres`.

# Fields

- `verbose::Bool`: Whether to throw a warning when the solver fails (defaults to `true`)
"""
Base.@kwdef struct IterativeLinearSolver <: AbstractLinearSolver
    verbose::Bool = true
end

presolve(::IterativeLinearSolver, A, y) = A

function solve(sol::IterativeLinearSolver, A, b)
    x, stats = gmres(A, b)
    if !stats.solved || stats.inconsistent
        sol.verbose && @warn "IterativeLinearSolver failed, result contains NaNs"
        x .= NaN
    end
    return x
end

"""
    DirectLinearSolver

An implementation of `AbstractLinearSolver` using the built-in backslash operator.

# Fields

- `verbose::Bool`: Whether to throw a warning when the solver fails (defaults to `true`)
"""
Base.@kwdef struct DirectLinearSolver <: AbstractLinearSolver
    verbose::Bool = true
end

function presolve(::DirectLinearSolver, A, y)
    return lu(Matrix(A); check=false)
end

function solve(sol::DirectLinearSolver, A_lu, b)
    x = A_lu \ b
    if !issuccess(A_lu)
        sol.verbose && @warn "DirectLinearSolver failed, result contains NaNs"
        x .= NaN
    end
    return x
end
