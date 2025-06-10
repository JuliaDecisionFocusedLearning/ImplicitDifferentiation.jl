## Linear solver

"""
    IterativeLinearSolver

Callable object that can solve linear systems `Ax = b` and `AX = B` in the same way as the built-in `\\`.

# Constructor

    IterativeLinearSolver{package}(; kwargs...)

The type parameter `package` can be either:

- `:Krylov` to use the solver `gmres` from [Krylov.jl](https://github.com/JuliaSmoothOptimizers/Krylov.jl)
- `:IterativeSolvers` to use the solver `gmres` from [IterativeSolvers.jl](https://github.com/JuliaLinearAlgebra/IterativeSolvers.jl)

Keyword arguments are passed on to the respective solver.

# Callable behavior

    (::IterativeLinearSolver)(A, b::AbstractVector)

Solve a linear system with a single right-hand side.

    (::IterativeLinearSolver)(A, B::AbstractMatrix)

Solve a linear system with multiple right-hand sides.
"""
struct IterativeLinearSolver{package,K}
    kwargs::K
    function IterativeLinearSolver{package}(; kwargs...) where {package}
        @assert package in [:Krylov, :IterativeSolvers]
        return new{package,typeof(kwargs)}(kwargs)
    end
end

IterativeLinearSolver() = IterativeLinearSolver{:Krylov}()

function (solver::IterativeLinearSolver{:Krylov})(A, b::AbstractVector)
    x, stats = Krylov.gmres(A, b; solver.kwargs...)
    return x
end

function (solver::IterativeLinearSolver{:Krylov})(A, B::AbstractMatrix)
    # TODO: use block_gmres
    X = mapreduce(hcat, eachcol(B)) do b
        x, _ = Krylov.gmres(A, b; solver.kwargs...)
        x
    end
    return X
end

function (solver::IterativeLinearSolver{:IterativeSolvers})(A, b::AbstractVector)
    x = IterativeSolvers.gmres(A, b; solver.kwargs...)
    return x
end

function (solver::IterativeLinearSolver{:IterativeSolvers})(A, B::AbstractMatrix)
    X = mapreduce(hcat, eachcol(B)) do b
        IterativeSolvers.gmres(A, b; solver.kwargs...)
    end
    return X
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

Specify that the matrix `A` involved in the implicit function theorem should be represented lazily.

# Constructors

    OperatorRepresentation{package}(; symmetric=false, hermitian=false)

The type parameter `package` can be either:

- `:LinearOperators` to use a wrapper from [LinearOperators.jl](https://github.com/JuliaSmoothOptimizers/LinearOperators.jl) (the default)
- `:LinearMaps` to use a wrapper from [LinearMaps.jl](https://github.com/JuliaLinearAlgebra/LinearMaps.jl)

The keyword arguments `symmetric` and `hermitian` give additional properties of the Jacobian of the `conditions` with respect to the solution `y`, in case you can prove them.

# See also

- [`ImplicitFunction`](@ref)
- [`MatrixRepresentation`](@ref)
"""
struct OperatorRepresentation{package,symmetric,hermitian} <: AbstractRepresentation
    function OperatorRepresentation{package}(;
        symmetric::Bool=false, hermitian::Bool=false
    ) where {package}
        @assert package in [:LinearOperators, :LinearMaps]
        return new{package,symmetric,hermitian}()
    end
end

OperatorRepresentation() = OperatorRepresentation{:LinearOperators}()

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
