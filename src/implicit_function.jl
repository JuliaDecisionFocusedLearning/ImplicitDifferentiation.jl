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
function (implicit::ImplicitFunction)(x; kwargs...)
    y, z = implicit.forward(x; kwargs...)
    return y, z
end

# Trick JET into thinking there exists an implementation for frule_via_ad
function ChainRulesCore.frule_via_ad(
    ::RuleConfig{>:HasForwardsMode}, ȧrgs, f, args...; kwargs...
)
    return nothing, nothing
end

"""
    frule(rc, (_, dx), implicit, x[; kwargs...])

Custom forward rule for [`ImplicitFunction{F,C,L}`](@ref).

We compute the Jacobian-vector product `Jv` by solving `Au = -Bv` and setting `Jv = u`.
Keyword arguments are given to both `implicit.forward` and `implicit.conditions`.
"""
function ChainRulesCore.frule(
    rc::RuleConfig, (_, dx), implicit::ImplicitFunction, x::AbstractArray{R}; kwargs...
) where {R<:Real}
    conditions = implicit.conditions
    linear_solver = implicit.linear_solver

    y, z = implicit(x; kwargs...)
    n, m = length(x), length(y)

    function pushforward_A(dỹ)
        dxyz = (NoTangent(), dỹ, ZeroTangent())
        F, dF = frule_via_ad(rc, dxyz, conditions, x, y, z; kwargs...)
        return dF
    end

    function pushforward_B(dx̃)
        dxyz = (dx̃, ZeroTangent(), ZeroTangent())
        F, dF = frule_via_ad(rc, dxyz, conditions, x, y, z; kwargs...)
        return dF
    end

    function mul_A!(res::Vector, dy_vec::Vector)
        dy = reshape(dy_vec, size(y))
        dF = pushforward_A(dy)
        return res .= vec(dF)
    end

    function mul_B!(res::Vector, dx_vec::Vector)
        dx = reshape(dx_vec, size(x))
        dF = pushforward_B(dx)
        return res .= vec(dF)
    end

    A = LinearOperator(R, m, m, false, false, mul_A!)
    B = LinearOperator(R, m, n, false, false, mul_B!)

    dx_vec = convert(Vector{R}, vec(unthunk(dx)))
    b = -B * dx_vec
    dy_vec, stats = linear_solver(A, b)
    if !stats.solved
        throw(SolverFailureException("Linear solver failed to converge", stats))
    end
    dy = reshape(dy_vec, size(y))
    dz = NoTangent()

    return (y, z), (dy, dz)
end

"""
    rrule(rc, implicit, x[; kwargs...])

Custom reverse rule for [`ImplicitFunction{F,C,L}`](@ref).

We compute the vector-Jacobian product `Jᵀv` by solving `Aᵀu = v` and setting `Jᵀv = -Bᵀu`.
Keyword arguments are given to both `implicit.forward` and `implicit.conditions`.
"""
function ChainRulesCore.rrule(
    rc::RuleConfig, implicit::ImplicitFunction, x::AbstractArray{R}; kwargs...
) where {R<:Real}
    conditions = implicit.conditions
    linear_solver = implicit.linear_solver

    y, z = implicit(x; kwargs...)
    n, m = length(x), length(y)

    _, pullback = rrule_via_ad(rc, conditions, x, y, z; kwargs...)

    function mul_Aᵀ!(res::Vector, dF_vec::Vector)
        dF = reshape(dF_vec, size(y))
        dconditions, dx, dy, dz = pullback(dF)
        return res .= vec(dy)
    end

    function mul_Bᵀ!(res::Vector, dF_vec::Vector)
        dF = reshape(dF_vec, size(y))
        dconditions, dx, dy, dz = pullback(dF)
        return res .= vec(dx)
    end

    Aᵀ = LinearOperator(R, m, m, false, false, mul_Aᵀ!)
    Bᵀ = LinearOperator(R, n, m, false, false, mul_Bᵀ!)

    function implicit_pullback((dy, dz))
        dy_vec = convert(Vector{R}, vec(unthunk(dy)))
        dF_vec, stats = linear_solver(Aᵀ, dy_vec)
        if !stats.solved
            throw(SolverFailureException("Linear solver failed to converge", stats))
        end
        dx_vec = -Bᵀ * dF_vec
        dx = reshape(dx_vec, size(x))
        return (NoTangent(), dx)
    end

    return (y, z), implicit_pullback
end
