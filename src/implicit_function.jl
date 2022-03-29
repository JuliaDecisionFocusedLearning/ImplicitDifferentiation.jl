"""
    ImplicitFunction{F,C,L}

Differentiable wrapper for an implicit function `x -> ŷ(x)` whose output is defined by explicit conditions `F(x,ŷ(x)) = 0`.

We can obtain the Jacobian of `ŷ` with the implicit function theorem:
```
∂₁F(x,ŷ(x)) + ∂₂F(x,ŷ(x)) * ∂ŷ(x) = 0
```
This amounts to solving the linear system `A * J = B`, where `A ∈ ℝᶜᵐ`, `B ∈ ℝᶜⁿ` and `J ∈ ℝᵐⁿ`.

# Fields:
- `forward::F`: callable of the form `x -> ŷ(x)`
- `conditions::C`: callable of the form `(x,y) -> F(x,y)`
- `linear_solver::L`: callable of the form `(A,b) -> u` such that `A * u = b`
"""
Base.@kwdef struct ImplicitFunction{F,C,L}
    forward::F
    conditions::C
    linear_solver::L
end

"""
    implicit(x)

Make [`ImplicitFunction`](@ref) callable by applying `implicit.forward`.
"""
(implicit::ImplicitFunction)(x) = implicit.forward(x)

"""
    frule(rc, (_, Δx), implicit, x)

Custom forward rule for [`ImplicitFunction`](@ref).
"""
function ChainRulesCore.frule(rc::RuleConfig, (_, Δx), implicit::ImplicitFunction, x)
    @unpack forward, conditions, linear_solver = implicit
    y = forward(x)
    Δy = nothing
    return y, Δy
end

"""
    rrule(rc, implicit, x)

Custom reverse rule for [`ImplicitFunction`](@ref).
"""
function ChainRulesCore.rrule(rc::RuleConfig, implicit::ImplicitFunction, x)
    @unpack forward, conditions, linear_solver = implicit
    y = forward(x)

    F₁(x̃) = conditions(x̃, y)
    F₂(ỹ) = -conditions(x, ỹ)

    _, pullback_Aᵀ = rrule_via_ad(rc, F₂, y)
    _, pullback_Bᵀ = rrule_via_ad(rc, F₁, x)

    n, m, c = length(x), length(y), length(y)
    Aᵀ = LinearMap(last ∘ pullback_Aᵀ, m, c)  # v -> Aᵀv
    Bᵀ = LinearMap(last ∘ pullback_Bᵀ, n, c)  # u -> Bᵀu

    function implicit_pullback(dy)
        u::Vector{Float64} = linear_solver(Aᵀ, unthunk(dy))  # u = At \ v
        dx::Vector{Float64} = Bᵀ * u
        return (NoTangent(), dx)
    end

    return y, implicit_pullback
end
