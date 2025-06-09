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
        representation=OperatorRepresentation{:LinearOperators}(),
        linear_solver=IterativeLinearSolver{:Krylov}(),
        backend=nothing,
        preparation=nothing,
        input_example=nothing,
    )

## Positional arguments

- `solver`: a callable returning `(x, args...) -> (y, z)` where `z` is an arbitrary byproduct of the solve. Both `x` and `y` must be subtypes of `AbstractArray`, while `z` and `args` can be anything.
- `conditions`: a callable returning a vector of optimality conditions `(x, y, z, args...) -> c`, must be compatible with automatic differentiation

## Keyword arguments

- `representation`: either [`MatrixRepresentation`](@ref) or [`OperatorRepresentation`](@ref)
- `linear_solver`: a callable to solve linear systems with two required methods, one for `(A, b)` (single solve) and one for `(A, B)` (batched solve). It defaults to [`IterativeLinearSolver`](@ref) but can also be the built-in `\\`, or a user-provided function.
- `backend::AbstractADType`: specifies how the `conditions` will be differentiated with respect to `x` and `y`. It can be either
    - `nothing`, which means that the external autodiff system will be used
    - a single object from [ADTypes.jl](https://github.com/SciML/ADTypes.jl)
    - a named tuple `(; x, y)` of objects from [ADTypes.jl](https://github.com/SciML/ADTypes.jl)
- `preparation`: either `nothing` or a mode object from [ADTypes.jl](https://github.com/SciML/ADTypes.jl): `ADTypes.ForwardMode()`, `ADTypes.ReverseMode()` or `ADTypes.ForwardOrReverseMode()`.
- `input_example`: either `nothing` or a tuple `(x, args...)` used to prepare differentiation.
- `strict::Val=Val(true)`: whether or not to enforce a strict match in [DifferentiationInterface.jl](https://github.com/JuliaDiff/DifferentiationInterface.jl) between the preparation and the execution types. Relaxing this to `strict=Val(false)` can prove necessary when working with custom array types like ComponentArrays.jl, which are not always compatible with iterative linear solvers.
"""
struct ImplicitFunction{
    F,
    C,
    L,
    R<:AbstractRepresentation,
    B<:Union{
        Nothing,  #
        AbstractADType,
        NamedTuple{(:x, :y),<:Tuple{AbstractADType,AbstractADType}},
    },
    P<:Union{Nothing,AbstractMode},
    PA,
    PAT,
    PB,
    PBT,
    _strict,
}
    solver::F
    conditions::C
    linear_solver::L
    representation::R
    backends::B
    preparation::P
    prep_A::PA
    prep_Aᵀ::PAT
    prep_B::PB
    prep_Bᵀ::PBT
    strict::Val{_strict}
end

function ImplicitFunction(
    solver,
    conditions;
    representation=OperatorRepresentation{:LinearOperators}(),
    linear_solver=IterativeLinearSolver{:Krylov}(),
    backends=nothing,
    preparation=nothing,
    input_example=nothing,
    strict::Val=Val(true),
)
    if isnothing(preparation) || isnothing(backends) || isnothing(input_example)
        prep_A = nothing
        prep_Aᵀ = nothing
        prep_B = nothing
        prep_Bᵀ = nothing
    else
        if backends isa AbstractADType
            backends = (backends, backends)
        end
        x, args = first(input_example), Base.tail(input_example)
        y, z = solver(x, args...)
        c = conditions(x, y, z, args...)
        if preparation isa Union{ForwardMode,ForwardOrReverseMode}
            prep_A = prepare_A(
                representation, x, y, z, c, args...; conditions, backend=backends.y, strict
            )
            prep_B = prepare_B(
                representation, x, y, z, c, args...; conditions, backend=backends.x, strict
            )
        else
            prep_A = nothing
            prep_B = nothing
        end
        if preparation isa Union{ReverseMode,ForwardOrReverseMode}
            prep_Aᵀ = prepare_Aᵀ(
                representation, x, y, z, c, args...; conditions, backend=backends.y, strict
            )
            prep_Bᵀ = prepare_Bᵀ(
                representation, x, y, z, c, args...; conditions, backend=backends.x, strict
            )
        else
            prep_Aᵀ = nothing
            prep_Bᵀ = nothing
        end
    end
    return ImplicitFunction(
        solver,
        conditions,
        linear_solver,
        representation,
        backends,
        preparation,
        prep_A,
        prep_Aᵀ,
        prep_B,
        prep_Bᵀ,
        strict,
    )
end

function Base.show(io::IO, implicit::ImplicitFunction)
    (; solver, conditions, backends, linear_solver, representation, preparation) = implicit
    return print(
        io,
        """
        ImplicitFunction(
            $solver,
            $conditions;
            representation=$representation,
            linear_solver=$linear_solver,
            backends=$backends,
            preparation=$preparation,
        )
        """,
    )
end

function (implicit::ImplicitFunction)(x::AbstractArray, args::Vararg{Any,N}) where {N}
    return implicit.solver(x, args...)
end
