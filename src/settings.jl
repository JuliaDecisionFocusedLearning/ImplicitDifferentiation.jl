## Linear solver

"""
    IterativeLinearSolver

Callable object that can solve linear systems `Ax = b` and `AX = B` in the same way as the built-in `\\`.

# Constructor

    IterativeLinearSolver(; kwargs...)
    IterativeLinearSolver{package}(; kwargs...)

The type parameter `package` can be either:

- `:Krylov` to use the solver `gmres` or `block_gmres` from [Krylov.jl](https://github.com/JuliaSmoothOptimizers/Krylov.jl) (the default)
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

function Base.show(io::IO, linear_solver::IterativeLinearSolver{package}) where {package}
    print(io, "IterativeLinearSolver{$(repr(package))}(; ")
    for (k, v) in pairs(linear_solver.kwargs)
        print(io, "$k=$v, ")
    end
    return print(io, ")")
end

IterativeLinearSolver(; kwargs...) = IterativeLinearSolver{:Krylov}(; kwargs...)

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

    OperatorRepresentation(;
        symmetric=false, hermitian=false, posdef=false, keep_input_type=false
    )
    OperatorRepresentation{package}(;
        symmetric=false, hermitian=false, posdef=false, keep_input_type=false
    )

The type parameter `package` can be either:

- `:LinearOperators` to use a wrapper from [LinearOperators.jl](https://github.com/JuliaSmoothOptimizers/LinearOperators.jl) (the default)
- `:LinearMaps` to use a wrapper from [LinearMaps.jl](https://github.com/JuliaLinearAlgebra/LinearMaps.jl)

The keyword arguments `symmetric`, `hermitian` and `posdef` give additional properties of the Jacobian of the `conditions` with respect to the solution `y`, which are useful to the solver in case you can prove them.

The keyword argument `keep_input_type` dictates whether to force the linear operator to work with the provided input type, or fall back on a default.

# See also

- [`ImplicitFunction`](@ref)
- [`MatrixRepresentation`](@ref)
"""
struct OperatorRepresentation{package,symmetric,hermitian,posdef,keep_input_type} <:
       AbstractRepresentation
    function OperatorRepresentation{package}(;
        symmetric::Bool=false,
        hermitian::Bool=false,
        posdef::Bool=false,
        keep_input_type::Bool=false,
    ) where {package}
        @assert package in [:LinearOperators, :LinearMaps]
        return new{package,symmetric,hermitian,posdef,keep_input_type}()
    end
end

function Base.show(
    io::IO, ::OperatorRepresentation{package,symmetric,hermitian,posdef,keep_input_type}
) where {package,symmetric,hermitian,posdef,keep_input_type}
    return print(
        io,
        "OperatorRepresentation{$(repr(package))}(; symmetric=$symmetric, hermitian=$hermitian, posdef=$posdef, keep_input_type=$keep_input_type)",
    )
end

OperatorRepresentation(; kwargs...) = OperatorRepresentation{:LinearOperators}(; kwargs...)

function chainrules_suggested_backend end
