"""
    ImplicitFunction{F,C,L}

Differentiable wrapper for an implicit function `x -> y(x)` whose output is defined by conditions `F(x,y(x)) = 0`.

More generally, we consider functions `x -> (y(x),z(x))` and conditions `F(x,y(x),z(x)) = 0`, where `z(x)` contains additional information that _is considered constant for differentiation purposes_. Beware: the method `zero(z)` must exist.

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
struct ImplicitFunction{F,C,L}
    forward::F
    conditions::C
    linear_solver::L
end

"""
    ImplicitFunction(forward, conditions)

Construct an `ImplicitFunction` with `Krylov.gmres` as the default linear solver.
"""
function ImplicitFunction(forward, conditions)
    return ImplicitFunction(forward, conditions, gmres)
end

"""
    implicit(x[; kwargs...])

Make `ImplicitFunction` callable by applying `implicit.forward`.
"""
function (implicit::ImplicitFunction)(x; kwargs...)
    y, z = implicit.forward(x; kwargs...)
    return y, z
end
