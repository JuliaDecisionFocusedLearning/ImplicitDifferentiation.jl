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
    frule(rc, (_, dx), implicit, x)

Custom forward rule for [`ImplicitFunction`](@ref).
"""
function ChainRulesCore.frule(rc::RuleConfig, (_, dx), implicit::ImplicitFunction, x)
    @unpack forward, conditions, linear_solver = implicit
    y = forward(x)

    F₁(x̃) = conditions(x̃, y)
    F₂(ỹ) = -conditions(x, ỹ)

    pushforward_A(dỹ) = last(frule_via_ad(rc, (NoTangent(), dỹ), F₂, y))
    pushforward_B(dx̃) = last(frule_via_ad(rc, (NoTangent(), dx̃), F₁, x))

    n, m, c = length(x), length(y), length(y)
    A = LinearMap(pushforward_A, c, m)
    B = LinearMap(pushforward_B, c, n)

    dy::Vector{Float64} = linear_solver(A, B * unthunk(dx))

    return y, dy
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

    pullback_Aᵀ = last ∘ rrule_via_ad(rc, F₂, y)[2]
    pullback_Bᵀ = last ∘ rrule_via_ad(rc, F₁, x)[2]

    n, m, c = length(x), length(y), length(y)
    Aᵀ = LinearMap(pullback_Aᵀ, m, c)
    Bᵀ = LinearMap(pullback_Bᵀ, n, c)

    function implicit_pullback(dy)
        u::Vector{Float64} = linear_solver(Aᵀ, unthunk(dy))
        dx::Vector{Float64} = Bᵀ * u
        return (NoTangent(), dx)
    end

    return y, implicit_pullback
end
