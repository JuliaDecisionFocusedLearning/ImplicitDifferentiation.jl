"""
    ImplicitFunction

Wrapper for an implicit function defined by a _solver_ and a set of _conditions_ which the solution satisfies.

An `ImplicitFunction` object behaves like a function, with the following signature:
    
    y, z = (implicit::ImplicitFunction)(x, args...)

The first output `y` is differentiable with respect to the first argument `x`, while the second output `z` (a byproduct of the solve) and the following positional arguments `args` are considered constant.

When a derivative is queried, the Jacobian of `y(x)` is computed using the implicit function theorem applied to the conditions `c(x, y)` (we ignore `z` for concision):

    ∂₂c(x, y(x)) * ∂y(x) = -∂₁c(x, y(x))

This requires solving a linear system `A * J = -B` where `A = ∂₂c`, `B = ∂₁c` and `J = ∂y`.

# Constructor

    ImplicitFunction(
        solver,
        conditions;
        representation=OperatorRepresentation(),
        linear_solver=IterativeLinearSolver(),
        backends=nothing,
        strict=Val(true),
    )

## Positional arguments

- `solver`: a callable returning `(x, args...) -> (y, z)` where `z` is an arbitrary byproduct of the solve. Both `x` and `y` must be subtypes of `AbstractArray`, while `z` and `args` can be anything.
- `conditions`: a callable returning a vector of optimality conditions `(x, y, z, args...) -> c`, must be compatible with automatic differentiation.

## Keyword arguments

- `representation`: defines how the partial Jacobian `A` of the conditions with respect to the output is represented. It can be either [`MatrixRepresentation`](@ref) or [`OperatorRepresentation`](@ref).
- `linear_solver`: specifies how the linear system `A * J = -B` will be solved in the implicit function theorem. It can be either [`DirectLinearSolver`](@ref) or [`IterativeLinearSolver`](@ref).
- `backends::AbstractADType`: specifies how the `conditions` will be differentiated with respect to `x` and `y`. It can be either, `nothing`, which means that the external autodiff system will be used, or a named tuple `(; x=AutoSomething(), y=AutoSomethingElse())` of backend objects from [ADTypes.jl](https://github.com/SciML/ADTypes.jl).
- `strict::Val`: specifies whether preparation inside [DifferentiationInterface.jl](https://github.com/JuliaDiff/DifferentiationInterface.jl) should enforce a strict match between the primal variables and the provided tangents.
"""
struct ImplicitFunction{
    F,
    C,
    L,
    R<:AbstractRepresentation,
    B<:Union{
        Nothing,  #
        NamedTuple{(:x, :y),<:Tuple{AbstractADType,AbstractADType}},
    },
    S,
}
    solver::F
    conditions::C
    linear_solver::L
    representation::R
    backends::B
    strict::Val{S}
end

function ImplicitFunction(
    solver,
    conditions;
    representation=OperatorRepresentation(),
    linear_solver=IterativeLinearSolver(),
    backends=nothing,
    strict::Val=Val(true),
)
    return ImplicitFunction(
        solver, conditions, linear_solver, representation, backends, strict
    )
end

function Base.show(io::IO, implicit::ImplicitFunction)
    (; solver, conditions, backends, linear_solver, representation) = implicit
    return print(
        io,
        """
        ImplicitFunction(
            $solver,
            $conditions;
            representation=$representation,
            linear_solver=$linear_solver,
            backends=$backends,
        )
        """,
    )
end
