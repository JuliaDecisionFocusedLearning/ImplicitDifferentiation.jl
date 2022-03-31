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

    pushforward_A(dỹ) = frule_via_ad(rc, (NoTangent(), dỹ), F₂, y)[2]
    pushforward_B(dx̃) = frule_via_ad(rc, (NoTangent(), dx̃), F₁, x)[2]

    n, m, c = length(x), length(y), length(y)
    mul_A!(res, v) = res .= pushforward_A(v)
    mul_B!(res, v) = res .= pushforward_B(v)

    A = LinearOperator(R, c, m, false, false, mul_A!)
    B = LinearOperator(R, c, m, false, false, mul_B!)

    b = B * unthunk(dx)
    dy, stats = linear_solver(A, b)

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

    pullback_Aᵀ = last ∘ rrule_via_ad(rc, F₂, y)[2]
    pullback_Bᵀ = last ∘ rrule_via_ad(rc, F₁, x)[2]

    n, m, c = length(x), length(y), length(y)
    mul_Aᵀ!(res, v) = res .= pullback_Aᵀ(v)
    mul_Bᵀ!(res, v) = res .= pullback_Bᵀ(v)

    Aᵀ = LinearOperator(R, m, c, false, false, mul_Aᵀ!)
    Bᵀ = LinearOperator(R, n, c, false, false, mul_Bᵀ!)

    function implicit_pullback(dy)
        u, stats = linear_solver(Aᵀ, unthunk(dy))
        dx = Bᵀ * u
        return (NoTangent(), dx)
    end

    return y, implicit_pullback
end
