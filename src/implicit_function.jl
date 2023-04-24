"""
    ImplicitFunction{F,C,L}

Differentiable wrapper for an implicit function `x -> ŷ(x)` whose output is defined by explicit conditions `F(x,ŷ(x)) = 0`.

If `x ∈ ℝⁿ` and `y ∈ ℝᵈ`, then we need as many conditions as output dimensions: `F(x,y) ∈ ℝᵈ`.
Thanks to these conditions, we can compute the Jacobian of `ŷ(⋅)` using the implicit function theorem:
```
∂₂F(x,ŷ(x)) * ∂ŷ(x) = -∂₁F(x,ŷ(x))
```
This requires solving a linear system `A * J = -B`, where `A ∈ ℝᵈˣᵈ`, `B ∈ ℝᵈˣⁿ` and `J ∈ ℝᵈˣⁿ`.

# Fields:
- `forward::F`: callable of the form `x -> (ŷ(x),z)`
- `conditions::C`: callable of the form `(x,y,z) -> F(x,y,z)`
- `linear_solver::L`: callable of the form `(A,b) -> u` such that `Au = b`
"""
struct ImplicitFunction{F,C,L}
    forward::F
    conditions::C
    linear_solver::L
end

"""
    ImplicitFunction(forward, conditions)

Construct an [`ImplicitFunction{F,C,L}`](@ref) with `Krylov.gmres` as the default linear solver.
"""
function ImplicitFunction(forward::F, conditions::C) where {F,C}
    return ImplicitFunction(forward, conditions, gmres)
end

"""
    implicit(x[; kwargs...])

Make [`ImplicitFunction{F,C,L}`](@ref) callable by applying `implicit.forward`.
"""
function (implicit::ImplicitFunction)(x; kwargs...)
    y, z = implicit.forward(x; kwargs...)
    return y, z
end
