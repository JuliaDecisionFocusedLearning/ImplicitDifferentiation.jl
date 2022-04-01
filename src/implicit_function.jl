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

struct SolverFailureException <: Exception
    msg::String
end

"""
    implicit(x)

Make [`ImplicitFunction`](@ref) callable by applying `implicit.forward`.
"""
(implicit::ImplicitFunction)(x) = implicit.forward(x)

"""
    frule(rc, (_, dx), implicit, x)

Custom forward rule for [`ImplicitFunction`](@ref).

We compute the Jacobian-vector product `Jv` by solving `Au = Bv` and setting `Jv = u`.
"""
function ChainRulesCore.frule(
    rc::RuleConfig, (_, dx), implicit::ImplicitFunction, x)
    forward = implicit.forward
    conditions = implicit.conditions
    linear_solver = implicit.linear_solver

    y = forward(x)

    x_vec, _ = flatten(x)
    y_vec, unflatten_y = flatten(y)
    n, m = length(x_vec), length(y_vec)

    F₁(x̃) = conditions(x̃, y)
    F₂(ỹ) = -conditions(x, ỹ)

    pushforward_A(dỹ) = frule_via_ad(rc, (NoTangent(), dỹ), F₂, y)[2]
    pushforward_B(dx̃) = frule_via_ad(rc, (NoTangent(), dx̃), F₁, x)[2]

    mul_A!(res, v) = res .= flatten(pushforward_A(v))[1]
    mul_B!(res, v) = res .= flatten(pushforward_B(v))[1]

    A = LinearOperator(Float64, m, m, false, false, mul_A!)
    B = LinearOperator(Float64, m, n, false, false, mul_B!)

    dx_vec = flatten(unthunk(dx))[1]
    b = B * dx_vec
    dy_vec, stats = linear_solver(A, b)
    if !stats.solved
        throw(SolverFailureException("The linear solver failed to converge"))
    end
    dy = unflatten_y(dy_vec)

    return y, dy
end

"""
    rrule(rc, implicit, x)

Custom reverse rule for [`ImplicitFunction`](@ref).

We compute the vector-Jacobian product `Jᵀv` by solving `Aᵀu = v` and setting `Jᵀv = Bᵀu`.
"""
function ChainRulesCore.rrule(
    rc::RuleConfig, implicit::ImplicitFunction, x)
    forward = implicit.forward
    conditions = implicit.conditions
    linear_solver = implicit.linear_solver

    y = forward(x)

    x_vec, unflatten_x = flatten(x)
    y_vec, _ = flatten(y)
    n, m = length(x_vec), length(y_vec)

    F₁(x̃) = conditions(x̃, y)
    F₂(ỹ) = -conditions(x, ỹ)

    pullback_Aᵀ = last ∘ rrule_via_ad(rc, F₂, y)[2]
    pullback_Bᵀ = last ∘ rrule_via_ad(rc, F₁, x)[2]

    mul_Aᵀ!(res, v) = res .= flatten(pullback_Aᵀ(v))[1]
    mul_Bᵀ!(res, v) = res .= flatten(pullback_Bᵀ(v))[1]

    Aᵀ = LinearOperator(Float64, m, m, false, false, mul_Aᵀ!)
    Bᵀ = LinearOperator(Float64, n, m, false, false, mul_Bᵀ!)

    function implicit_pullback(dy)
        dy_vec = flatten(unthunk(dy))[1]
        u, stats = linear_solver(Aᵀ, dy_vec)
        if !stats.solved
            throw(SolverFailureException("The linear solver failed to converge"))
        end
        dx_vec = Bᵀ * u
        dx = unflatten_x(dx_vec)
        return (NoTangent(), dx)
    end

    return y, implicit_pullback
end
