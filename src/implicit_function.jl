"""
    ImplicitFunction{F,C,L}

Differentiable wrapper for an implicit function `x -> ŷ(x)` whose output is defined by explicit conditions `F(x,ŷ(x)) = 0`.

If `x ∈ ℝⁿ` and `y ∈ ℝᵈ`, then we need as many conditions as output dimensions: `F(x,y) ∈ ℝᵈ`.
Thanks to these conditions, we can compute the Jacobian of `ŷ(⋅)` using the implicit function theorem:
```
∂₂F(x,ŷ(x)) * ∂ŷ(x) = -∂₁F(x,ŷ(x))
```
This requires solving a linear system `A * J = B`, where `A ∈ ℝᵈˣᵈ`, `B ∈ ℝᵈˣⁿ` and `J ∈ ℝᵈˣⁿ`.

# Fields:
- `forward::F`: callable of the form `x -> ŷ(x)`
- `conditions::C`: callable of the form `(x,y) -> F(x,y)`
- `linear_solver::L`: callable of the form `(A,b) -> u` such that `A * u = b`
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

struct SolverFailureException <: Exception
    msg::String
end

"""
    implicit(x)

Make [`ImplicitFunction{F,C,L}`](@ref) callable by applying `implicit.forward`.
"""
(implicit::ImplicitFunction)(x; kwargs...) = implicit.forward(x; kwargs...)

"""
    frule(rc, (_, dx), implicit, x)

Custom forward rule for [`ImplicitFunction{F,C,L}`](@ref).

We compute the Jacobian-vector product `Jv` by solving `Au = Bv` and setting `Jv = u`.
"""
function ChainRulesCore.frule(
    rc::RuleConfig, (_, dx), implicit::ImplicitFunction, x::AbstractArray{R}; kwargs...
) where {R<:Real}
    forward = implicit.forward
    conditions = implicit.conditions
    linear_solver = implicit.linear_solver

    y = forward(x; kwargs...)

    conditions_x(x̃) = conditions(x̃, y)
    conditions_y(ỹ) = -conditions(x, ỹ)

    pushforward_A(dỹ) = frule_via_ad(rc, (NoTangent(), dỹ), conditions_y, y)[2]
    pushforward_B(dx̃) = frule_via_ad(rc, (NoTangent(), dx̃), conditions_x, x)[2]

    mul_A!(res::Vector, u::Vector) = res .= vec(pushforward_A(reshape(u, size(y))))
    mul_B!(res::Vector, v::Vector) = res .= vec(pushforward_B(reshape(v, size(x))))

    n, m = length(x), length(y)
    A = LinearOperator(R, m, m, false, false, mul_A!)
    B = LinearOperator(R, m, n, false, false, mul_B!)

    dx_vec = convert(Vector{R}, vec(unthunk(dx)))
    b = B * dx_vec
    dy_vec, stats = linear_solver(A, b)
    stats.solved || throw(SolverFailureException("Linear solver failed to converge"))
    dy = reshape(dy_vec, size(y))

    return y, dy
end

"""
    rrule(rc, implicit, x)

Custom reverse rule for [`ImplicitFunction{F,C,L}`](@ref).

We compute the vector-Jacobian product `Jᵀv` by solving `Aᵀu = v` and setting `Jᵀv = Bᵀu`.
"""
function ChainRulesCore.rrule(
    rc::RuleConfig, implicit::ImplicitFunction, x::AbstractArray{R}; kwargs...
) where {R<:Real}
    forward = implicit.forward
    conditions = implicit.conditions
    linear_solver = implicit.linear_solver

    y = forward(x; kwargs...)

    conditions_x(x̃) = conditions(x̃, y)
    conditions_y(ỹ) = -conditions(x, ỹ)

    pullback_Aᵀ = last ∘ rrule_via_ad(rc, conditions_y, y)[2]
    pullback_Bᵀ = last ∘ rrule_via_ad(rc, conditions_x, x)[2]

    mul_Aᵀ!(res::Vector, u::Vector) = res .= vec(pullback_Aᵀ(reshape(u, size(y))))
    mul_Bᵀ!(res::Vector, v::Vector) = res .= vec(pullback_Bᵀ(reshape(v, size(y))))

    n, m = length(x), length(y)
    Aᵀ = LinearOperator(R, m, m, false, false, mul_Aᵀ!)
    Bᵀ = LinearOperator(R, n, m, false, false, mul_Bᵀ!)

    function implicit_pullback(dy)
        dy_vec = convert(Vector{R}, vec(unthunk(dy)))
        u, stats = linear_solver(Aᵀ, dy_vec)
        stats.solved || throw(SolverFailureException("Linear solver failed to converge"))
        dx_vec = Bᵀ * u
        dx = reshape(dx_vec, size(x))
        return (NoTangent(), dx)
    end

    return y, implicit_pullback
end
