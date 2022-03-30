"""
    ImplicitFunction{F,C,L}

Differentiable wrapper for an implicit function `x -> ŷ(x)` whose output is defined by explicit conditions `F(x,ŷ(x)) = 0`.

We can obtain the Jacobian of `ŷ` with the implicit function theorem:
```
∂₁F(x,ŷ(x)) + ∂₂F(x,ŷ(x)) * ∂ŷ(x) = 0
```
If `x ∈ ℝⁿ`, `y ∈ ℝᵐ` and `F(x,y) ∈ ℝᶜ`, this amounts to solving the linear system `A * J = B`, where `A ∈ ℝᶜᵐ`, `B ∈ ℝᶜⁿ` and `J ∈ ℝᵐⁿ`.

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
(implicit::ImplicitFunction)(x::AbstractVector{<:Real}) = implicit.forward(x)

"""
    frule(rc, (_, dx), implicit, x)

Custom forward rule for [`ImplicitFunction`](@ref).

We compute the Jacobian-vector product `Jv` by solving `Au = Bv` and setting `Jv = u`.
"""
function ChainRulesCore.frule(rc::RuleConfig, (_, dx), implicit::ImplicitFunction, x::AbstractVector{R}) where {R<:Real}
    forward = implicit.forward
    conditions = implicit.conditions
    linear_solver = implicit.linear_solver

    y = forward(x)

    F₁(x̃) = conditions(x̃, y)
    F₂(ỹ) = -conditions(x, ỹ)

    pushforward_A(dỹ) = last(frule_via_ad(rc, (NoTangent(), dỹ), F₂, y))
    pushforward_B(dx̃) = last(frule_via_ad(rc, (NoTangent(), dx̃), F₁, x))

    n, m, c = length(x), length(y), length(y)
    A = LinearMap(pushforward_A, c, m)
    b::Vector{R} = pushforward_B(unthunk(dx))

    dy::Vector{R} = linear_solver(A, b)

    return y, dy
end

"""
    rrule(rc, implicit, x)

Custom reverse rule for [`ImplicitFunction`](@ref).

We compute the vector-Jacobian product `Jᵀv` by solving `Aᵀu = v` and setting `Jᵀv = Bᵀu`.
"""
function ChainRulesCore.rrule(rc::RuleConfig, implicit::ImplicitFunction, x::AbstractVector{R}) where {R<:Real}
    forward = implicit.forward
    conditions = implicit.conditions
    linear_solver = implicit.linear_solver

    y = forward(x)

    F₁(x̃) = conditions(x̃, y)
    F₂(ỹ) = -conditions(x, ỹ)

    pullback_Aᵀ = rrule_via_ad(rc, F₂, y)[2]
    pullback_Bᵀ = rrule_via_ad(rc, F₁, x)[2]

    n, m, c = length(x), length(y), length(y)
    Aᵀ = LinearMap(last ∘ pullback_Aᵀ, m, c)
    Bᵀ = LinearMap(last ∘ pullback_Bᵀ, n, c)

    function implicit_pullback(dy)
        u::Vector{R} = linear_solver(Aᵀ, unthunk(dy))
        dx::Vector{R} = Bᵀ * u
        return (NoTangent(), dx)
    end

    return y, implicit_pullback
end
