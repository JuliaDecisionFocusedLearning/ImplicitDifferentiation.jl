"""
    AbstractLinearSolver

All linear solvers used within an `ImplicitFunction` must satisfy this interface.

It can be useful to roll out your own solver if you need more fine-grained control on convergence / speed / behavior in case of singularity.
Check out the source code of `IterativeLinearSolver` and `DirectLinearSolver` for implementation examples. 

# Required methods

- `presolve(linear_solver, A, y)`: Returns a matrix-like object `A` for which it is cheaper to solve several linear systems with different vectors `b` (a typical example would be to perform LU factorization).
- `solve(linear_solver, A, b)`: Returns a vector `x` satisfying `Ax = b`. If the linear system has not been solved to satisfaction, every element of `x` should be a `NaN` of the appropriate floating point type.
"""
abstract type AbstractLinearSolver end

"""
    IterativeLinearSolver

An implementation of `AbstractLinearSolver` using `Krylov.gmres`.
"""
struct IterativeLinearSolver <: AbstractLinearSolver end

presolve(::IterativeLinearSolver, A, y) = A

function solve(::IterativeLinearSolver, A, b)
    T = float(promote_type(eltype(A), eltype(b)))
    x, stats = gmres(A, b)
    x_maybenan = similar(x, T)
    if stats.solved && !stats.inconsistent
        x_maybenan .= x
    else
        @warn "IterativeLinearSolver failed, result contains NaNs"
        x_maybenan .= convert(T, NaN)
    end
    return x_maybenan
end

"""
    DirectLinearSolver

An implementation of `AbstractLinearSolver` using the built-in backslash operator.
"""
struct DirectLinearSolver <: AbstractLinearSolver end

function presolve(::DirectLinearSolver, A, y)
    return lu(Matrix(A); check=false)
end

function solve(::DirectLinearSolver, A_lu, b)
    # workaround for https://github.com/JuliaArrays/StaticArrays.jl/issues/1190
    T = float(promote_type(eltype(A_lu.L), eltype(A_lu.U), eltype(b)))
    x_maybenan = Vector{T}(undef, size(A_lu.L, 2))
    if issuccess(A_lu)
        x_maybenan .= A_lu \ b
    else
        @warn "DirectLinearSolver failed, result contains NaNs"
        x_maybenan .= convert(T, NaN)
    end
    return x_maybenan
end
