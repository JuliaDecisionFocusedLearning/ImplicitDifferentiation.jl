## Linear solver

"""
    DirectLinearSolver

Specify that linear systems `Ax = b` should be solved with a direct method.
"""
struct DirectLinearSolver end

function (solver::DirectLinearSolver)(x::AbstractVector, A, b::AbstractVector)
    ldiv!(x, A, b)
    return x
end

"""
    IterativeLinearSolver

Specify that linear systems `Ax = b` should be solved with an iterative method.

# Constructor

    IterativeLinearSolver(::Val{method}=Val(:gmres); kwargs...)

The `method` symbol is used to pick the appropriate algorithm from [Krylov.jl](https://github.com/JuliaSmoothOptimizers/Krylov.jl).
Keyword arguments are passed on to that algorithm.
"""
struct IterativeLinearSolver{method,K}
    _method::Val{method}
    kwargs::K
    function IterativeLinearSolver((::Val{method})=Val(:gmres); kwargs...) where {method}
        return new{method,typeof(kwargs)}(Val(method), kwargs)
    end
end

function Base.show(io::IO, linear_solver::IterativeLinearSolver{method}) where {method}
    print(io, "IterativeLinearSolver{$(repr(method))}")
    if isempty(linear_solver.kwargs)
        print(io, "()")
    else
        print(io, "(; ")
        for (k, v) in pairs(linear_solver.kwargs)
            print(io, "$k=$(repr(v)), ")
        end
        print(io, ")")
    end
end

function (solver::IterativeLinearSolver{method})(
    x::AbstractVector, A, b::AbstractVector
) where {method}
    if typeof(b) == typeof(x)
        constructor = KrylovConstructor(b, x)
        workspace = krylov_workspace(Val(method), constructor)
    else
        workspace = krylov_workspace(Val(method), A, b)
    end
    krylov_solve!(workspace, A, b)
    copyto!(x, solution(workspace))
    return x
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

    OperatorRepresentation(; symmetric=false, hermitian=false, posdef=false)
    OperatorRepresentation{package}(; symmetric=false, hermitian=false, posdef=false)

The type parameter `package` can be either:

- `:LinearOperators` to use a wrapper from [LinearOperators.jl](https://github.com/JuliaSmoothOptimizers/LinearOperators.jl) (the default)
- `:LinearMaps` to use a wrapper from [LinearMaps.jl](https://github.com/JuliaLinearAlgebra/LinearMaps.jl)

The keyword arguments `symmetric`, `hermitian` and `posdef` give additional properties of the Jacobian of the `conditions` with respect to the solution `y`, which are useful to the solver in case you can prove them.

# See also

- [`ImplicitFunction`](@ref)
- [`MatrixRepresentation`](@ref)
"""
struct OperatorRepresentation{package,symmetric,hermitian,posdef} <: AbstractRepresentation
    function OperatorRepresentation{package}(;
        symmetric::Bool=false, hermitian::Bool=false, posdef::Bool=false
    ) where {package}
        @assert package in [:LinearOperators, :LinearMaps]
        return new{package,symmetric,hermitian,posdef}()
    end
end

function Base.show(
    io::IO, ::OperatorRepresentation{package,symmetric,hermitian,posdef}
) where {package,symmetric,hermitian,posdef}
    return print(
        io,
        "OperatorRepresentation{$(repr(package))}(; symmetric=$symmetric, hermitian=$hermitian, posdef=$posdef)",
    )
end

OperatorRepresentation(; kwargs...) = OperatorRepresentation{:LinearOperators}(; kwargs...)

function chainrules_suggested_backend end
