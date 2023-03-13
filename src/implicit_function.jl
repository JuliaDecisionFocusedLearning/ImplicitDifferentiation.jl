"""
    ImplicitFunction{F,C,L}

Differentiable wrapper for an implicit function `x -> ŷ(x)` whose output is defined by explicit conditions `F(x,ŷ(x)) = 0`.

If `x ∈ ℝⁿ` and `y ∈ ℝᵈ`, then we need as many conditions as output dimensions: `F(x,y) ∈ ℝᵈ`.
Thanks to these conditions, we can compute the Jacobian of `ŷ(⋅)` using the implicit function theorem:
```
-∂₂F(x,ŷ(x)) * ∂ŷ(x) = ∂₁F(x,ŷ(x))
```
This requires solving a linear system `-A * J = B`, where `A ∈ ℝᵈˣᵈ`, `B ∈ ℝᵈˣⁿ` and `J ∈ ℝᵈˣⁿ`.

# Fields:
- `forward::F`: callable of the form `x -> ŷ(x)`
- `conditions::C`: callable of the form `(x,y) -> F(x,y)`
- `linear_solver::L`: callable of the form `(C,D) -> U` such that `C * U = D`
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

struct SolverFailureException{S} <: Exception
    msg::String
    stats::S
end

function Base.show(io::IO, sfe::SolverFailureException)
    return println(io, "SolverFailureException: $(sfe.msg) \n Solver stats: $(sfe.stats)")
end

"""
    implicit(x[; kwargs...])

Make [`ImplicitFunction{F,C,L}`](@ref) callable by applying `implicit.forward`.
"""
(implicit::ImplicitFunction)(x; kwargs...) = implicit.forward(x; kwargs...)

"""
    frule(rc, (_, dx), implicit, x[; kwargs...])

Custom forward rule for [`ImplicitFunction{F,C,L}`](@ref).

We compute the Jacobian-vector product `Jv` by solving `Au = Bv` and setting `Jv = u`.
Keyword arguments are given to both `implicit.forward` and `implicit.conditions`.
"""
function ChainRulesCore.frule(
    rc::RuleConfig, (_, dx), implicit::ImplicitFunction, x::AbstractArray{R}; kwargs...
) where {R<:Real}
    conditions = implicit.conditions
    linear_solver = implicit.linear_solver

    y = implicit(x; kwargs...)

    function mul_A!(res::Vector, u::Vector)
        dy = reshape(u, size(y))
        dargs = (NoTangent(), ZeroTangent(), dy)
        dcond = frule_via_ad(rc, dargs, conditions, x, y; kwargs...)[2]
        return res .= vec(dcond)
    end

    function mul_B!(res::Vector, v::Vector)
        dx = reshape(v, size(x))
        dargs = (NoTangent(), dx, ZeroTangent())
        dcond = frule_via_ad(rc, dargs, conditions, x, y; kwargs...)[2]
        return res .= vec(dcond)
    end

    n, m = length(x), length(y)
    A = LinearOperator(R, m, m, false, false, mul_A!)
    B = LinearOperator(R, m, n, false, false, mul_B!)

    dx_vec = convert(Vector{R}, vec(unthunk(dx)))
    b = B * dx_vec
    dy_vec, stats = linear_solver(-A, b)
    if !stats.solved
        throw(SolverFailureException("Linear solver failed to converge", stats))
    end
    dy = reshape(dy_vec, size(y))

    return y, dy
end

"""
    rrule(rc, implicit, x[; kwargs...])

Custom reverse rule for [`ImplicitFunction{F,C,L}`](@ref).

We compute the vector-Jacobian product `Jᵀv` by solving `Aᵀu = v` and setting `Jᵀv = Bᵀu`.
Keyword arguments are given to both `implicit.forward` and `implicit.conditions`.
"""
function ChainRulesCore.rrule(
    rc::RuleConfig, implicit::ImplicitFunction, x::AbstractArray{R}; kwargs...
) where {R<:Real}
    conditions = implicit.conditions
    linear_solver = implicit.linear_solver

    y = implicit(x; kwargs...)

    pullback = rrule_via_ad(rc, conditions, x, y; kwargs...)[2]

    mul_Aᵀ!(res::Vector, u::Vector) = res .= vec(pullback(reshape(u, size(y)))[3])
    mul_Bᵀ!(res::Vector, v::Vector) = res .= vec(pullback(reshape(v, size(y)))[2])

    n, m = length(x), length(y)
    Aᵀ = LinearOperator(R, m, m, false, false, mul_Aᵀ!)
    Bᵀ = LinearOperator(R, n, m, false, false, mul_Bᵀ!)

    function implicit_pullback(dy)
        dy_vec = convert(Vector{R}, vec(unthunk(dy)))
        u, stats = linear_solver(-Aᵀ, dy_vec)
        if !stats.solved
            throw(SolverFailureException("Linear solver failed to converge", stats))
        end
        dx_vec = Bᵀ * u
        dx = reshape(dx_vec, size(x))
        return (NoTangent(), dx)
    end

    return y, implicit_pullback
end
