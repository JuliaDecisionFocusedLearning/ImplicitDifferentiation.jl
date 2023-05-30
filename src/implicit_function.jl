struct Forward{byproduct,F}
    f::F
    function Forward{byproduct}(f::F) where {byproduct,F}
        return new{byproduct,F}(f)
    end
end
(f::Forward{true})(x; kwargs...) = f.f(x; kwargs...)
(f::Forward{false})(x; kwargs...) = (f.f(x; kwargs...), 0)

struct Conditions{byproduct,F}
    f::F
    function Conditions{byproduct}(f::F) where {byproduct,F}
        return new{byproduct,F}(f)
    end
end
(f::Conditions{true})(x, y, z; kwargs...) = f.f(x, y, z; kwargs...)
(f::Conditions{false})(x, y, z; kwargs...) = f.f(x, y; kwargs...)

"""
    ImplicitFunction{F,C,L}

Differentiable wrapper for an implicit function `x -> y(x)` whose output is defined by conditions `F(x,y(x)) = 0`.

More generally, we consider functions `x -> (y(x),z(x))` and conditions `F(x,y(x),z(x)) = 0`, where `z(x)` is a byproduct _considered constant for differentiation purposes_.
If `x ∈ ℝⁿ` and `y ∈ ℝᵈ`, then we need as many conditions as output dimensions: `F(x,y,z) ∈ ℝᵈ`. Thanks to these conditions, we can compute the Jacobian of `y(⋅)` using the implicit function theorem:
```
∂₂F(x,y(x),z(x)) * ∂y(x) = -∂₁F(x,y(x),z(x))
```
This amounts to solving a linear system `A * J = -B`, where `A ∈ ℝᵈˣᵈ`, `B ∈ ℝᵈˣⁿ` and `J ∈ ℝᵈˣⁿ`.

# Fields:
- `forward::F`: callable of the form `x -> (ŷ(x),z(x))`.
- `conditions::C`: callable of the form `(x,y,z) -> F(x,y,z)`
- `linear_solver::L`: callable of the form `(A,b) -> u` such that `Au = b`, must be taken from Krylov.jl
"""
struct ImplicitFunction{byproduct,F<:Forward{byproduct},C<:Conditions{byproduct},L}
    forward::F
    conditions::C
    linear_solver::L
end

"""
    ImplicitFunction(forward, conditions, linear_solver, ::Val{byproduct} = Val(false))

Construct an `ImplicitFunction` with `linear_solver` as the linear solver used in the implicit
differentiation. By default, `ImplicitFunction(forward, conditions, linear_solver)` will assume
that the `forward` function returns a single output and that the `conditions` function accepts
two arguments.

In some cases, it may be that a byproduct of the `forward` function can be used in the
`conditions` function to save computation. To tell `ImplicitDifferentiation` that the
`forward` function returns a second constant byproduct output and that the `conditions`
function accepts that byproduct as a third argument, you can construct the implicit function
using `ImplicitFunction(forward, conditions, linear_solver, Val(true))`.
"""
function ImplicitFunction(
    forward, conditions, linear_solver, ::Val{byproduct}=Val(false)
) where {byproduct}
    _forward = Forward{byproduct}(forward)
    _conditions = Conditions{byproduct}(conditions)
    return ImplicitFunction(_forward, _conditions, linear_solver)
end

# JET complained without this method
function ImplicitFunction(
    forward::Forward, conditions::Conditions, ::Val{byproduct}
) where {byproduct}
    return ImplicitFunction(forward, conditions, gmres)
end

"""
    ImplicitFunction(forward, conditions, ::Val{byproduct} = Val(false))

Construct an `ImplicitFunction` with `Krylov.gmres` as the default linear solver.
By default, `ImplicitFunction(forward, conditions)` will assume that the `forward`
function returns a single output and that the `conditions` function accepts two arguments.

In some cases, it may be that a byproduct of the `forward` function can be used in the
`conditions` function to save computation. To tell `ImplicitDifferentiation` that the
`forward` function returns a second constant byproduct output and that the `conditions`
function accepts that byproduct as a third argument, you can construct the implicit function
using `ImplicitFunction(forward, conditions, Val(true))`.
"""
function ImplicitFunction(
    forward, conditions, ::Val{byproduct}=Val(false)
) where {byproduct}
    _forward = Forward{byproduct}(forward)
    _conditions = Conditions{byproduct}(conditions)
    return ImplicitFunction(_forward, _conditions, Val(byproduct))
end

"""
    implicit(x[; kwargs...])
    implicit(x, Val(true), [; kwargs...])

Make `ImplicitFunction` callable by applying `implicit.forward`.

The first (default) call signature only returns `y(x)`, while the second returns `(y(x), z(x))`.
"""
function (implicit::ImplicitFunction)(
    x, ::Val{byproduct}=Val(false); kwargs...
) where {byproduct}
    y, z = implicit.forward(x; kwargs...)
    return byproduct ? (y, z) : y
end
