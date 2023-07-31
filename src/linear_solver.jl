"""
    AbstractLinearSolver

All linear solvers used within an `ImplicitFunction` must satisfy this interface.

# Required methods

- `presolve(linear_solver, A, y)`: return a matrix-like object `A` for which it is cheaper to solve several linear systems with different vectors `b` (a typical example would be to perform LU factorization).
- `solve(linear_solver, A, b)`: return a tuple `(x, stats)` where `x` satisfies `Ax = b` and `stats.solved ∈ {true, false}`.
"""
abstract type AbstractLinearSolver end

"""
    IterativeLinearSolver

An implementation of `AbstractLinearSolver` using `Krylov.gmres`.
"""
struct IterativeLinearSolver <: AbstractLinearSolver end

presolve(::IterativeLinearSolver, A, y) = A

function solve(::IterativeLinearSolver, A, b)
    x, stats = gmres(A, b)
    if !stats.solved
        throw(SolverFailureException(gmres, stats))
    end
    return x
end

"""
    DirectLinearSolver

An implementation of `AbstractLinearSolver` using the built-in backslash operator.
"""
struct DirectLinearSolver <: AbstractLinearSolver end

presolve(::DirectLinearSolver, A, y) = lu(Matrix(A))
solve(::DirectLinearSolver, A, b) = A \ b

struct SolverFailureException{A,B} <: Exception
    solver::A
    stats::B
end

function Base.show(io::IO, sfe::SolverFailureException)
    return println(
        io,
        "SolverFailureException: \n Linear solver: $(sfe.solver) \n Solver stats: $(string(sfe.stats))",
    )
end
