"""
    ImplicitFunction

Wrapper for an implicit function defined by a solver and a set of conditions which the solution satisfies.

An `ImplicitFunction` object behaves like a function, with the following signature:
    
    y, z = (implicit::ImplicitFunction)(x, args...)

The first output `y` is differentiable with respect to the first argument `x`, while the second output `z` and subsequent positional arguments are considered constant.
Both `x` and `y` must be subtypes of `AbstractVector`, while `z` can be any byproduct of the solve.

When a derivative is queried, the Jacobian of `y(x)` is computed using the implicit function theorem applied to the conditions `c(x, y)` (we ignore `z` for concision):

    ∂₂c(x, y(x)) * ∂y(x) = -∂₁c(x, y(x))

This requires solving a linear system `A * J = -B` where `A = ∂₂c`, `B = ∂₁c` and `J = ∂y`.

# Constructor

    ImplicitFunction(
        solver,
        conditions;
        backend,
        input_example,
        linear_solver=KrylovLinearSolver(),
        lazy=true,
    )

- `solver`: a callable returning `(x, args...) -> (y, z)` where `z` is an arbitrary byproduct of the solve
- `conditions`: a callable returning a vector of optimality conditions `(x, y, z, args...) -> c`, must be compatible with automatic differentiation
- `backend`: an `AbstractADType` object from [ADTypes.jl](https://github.com/SciML/ADTypes.jl) dictating how how the conditions will be differentiated
- `input_example`: a tuple `(x, args...)` used to prepare differentiation
- `linear_solver`: a callable to solve linear systems with two methods, one for `(A, b)` (single solve) and one for `(A, B)` (batched solve)
- `lazy`: whether to represent `A` and `B` with a `LinearOperator` from [LinearOperators.jl](https://github.com/JuliaSmoothOptimizers/LinearOperators.jl) (`lazy = true`) or a materialized Jacobian matrix (`lazy = false`)
"""
struct ImplicitFunction{lazy,F,C,B,L,PA1,PA2,PB1,PB2}
    solver::F
    conditions::C
    backend::B
    linear_solver::L
    prep_A::PA1
    prep_Aᵀ::PA2
    prep_B::PB1
    prep_Bᵀ::PB2
end

function ImplicitFunction(
    solver::F,
    conditions::C;
    backend::B,
    input_example,
    linear_solver::L=KrylovLinearSolver(),
    lazy=true,
) where {F,C,B,L}
    # preparation
    x, args = first(input_example), Base.tail(input_example)
    y, z = solver(x, args...)
    prep_A = prepare_A(conditions, backend, x, y, z, args...; lazy)
    prep_Aᵀ = prepare_Aᵀ(conditions, backend, x, y, z, args...; lazy)
    prep_B = prepare_B(conditions, backend, x, y, z, args...; lazy)
    prep_Bᵀ = prepare_Bᵀ(conditions, backend, x, y, z, args...; lazy)
    return ImplicitFunction{
        lazy,F,C,B,L,typeof(prep_A),typeof(prep_Aᵀ),typeof(prep_B),typeof(prep_Bᵀ)
    }(
        solver, conditions, backend, linear_solver, prep_A, prep_Aᵀ, prep_B, prep_Bᵀ
    )
end

function Base.show(io::IO, implicit::ImplicitFunction{lazy}) where {lazy}
    (; solver, conditions, linear_solver, backend) = implicit
    return print(
        io, "ImplicitFunction{$lazy}($solver, $conditions, $backend, $linear_solver)"
    )
end

function (implicit::ImplicitFunction)(x::AbstractVector, args::Vararg{Any,N}) where {N}
    return implicit.solver(x, args...)
end
