module ImplicitDifferentiationChainRulesExt

using AbstractDifferentiation: ReverseRuleConfigBackend, pullback_function
using ChainRulesCore: ChainRulesCore, NoTangent, RuleConfig, ZeroTangent, rrule, unthunk
using ImplicitDifferentiation:
    ImplicitFunction, PullbackMul!, ReturnByproduct, presolve, solve
using LinearAlgebra: lmul!, mul!
using LinearOperators: LinearOperator
using SimpleUnPack: @unpack

"""
    rrule(rc, implicit, x; kwargs...)
    rrule(rc, implicit, x, ReturnByproduct(); kwargs...)

Custom reverse rule for an [`ImplicitFunction`](@ref), to ensure compatibility with reverse mode autodiff.

This is only available if ChainRulesCore.jl is loaded (extension), except on Julia < 1.9 where it is always available.

- By default, this returns a single output `y(x)` with a pullback accepting a single cotangent `dy`.
- If `ReturnByproduct()` is passed as an argument, this returns a couple of outputs `(y(x),z(x))` with a pullback accepting a couple of cotangents `(dy, dz)` (remember that `z(x)` is not differentiated so its cotangent is ignored).

We compute the vector-Jacobian product `Jᵀv` by solving `Aᵀu = v` and setting `Jᵀv = -Bᵀu` (see [`ImplicitFunction`](@ref) for the definition of `A` and `B`).
Keyword arguments are given to both `implicit.forward` and `implicit.conditions`.
"""
function ChainRulesCore.rrule(
    rc::RuleConfig,
    implicit::ImplicitFunction,
    x::AbstractArray{R},
    ::ReturnByproduct;
    kwargs...,
) where {R}
    @unpack conditions, linear_solver = implicit

    y, z = implicit(x, ReturnByproduct(); kwargs...)
    n, m = length(x), length(y)

    backend = ReverseRuleConfigBackend(rc)
    pbA = pullback_function(backend, _y -> conditions(x, _y, z; kwargs...), y)
    pbB = pullback_function(backend, _x -> conditions(_x, y, z; kwargs...), x)
    pbmA = PullbackMul!(pbA, size(y))
    pbmB = PullbackMul!(pbB, size(y))

    Aᵀ_op = LinearOperator(R, m, m, false, false, pbmA)
    Bᵀ_op = LinearOperator(R, n, m, false, false, pbmB)
    Aᵀ_op_presolved = presolve(linear_solver, Aᵀ_op, y)

    implicit_pullback = ImplicitPullback(Aᵀ_op_presolved, Bᵀ_op, linear_solver, x)

    return (y, z), implicit_pullback
end

function ChainRulesCore.rrule(
    rc::RuleConfig, implicit::ImplicitFunction, x::AbstractArray{R}; kwargs...
) where {R}
    (y, z), implicit_pullback = rrule(rc, implicit, x, ReturnByproduct(); kwargs...)
    implicit_pullback_no_byproduct(dy) = Base.front(implicit_pullback((dy, nothing)))
    return y, implicit_pullback_no_byproduct
end

struct ImplicitPullback{A,B,L,X}
    Aᵀ_op::A
    Bᵀ_op::B
    linear_solver::L
    x::X
end

function (implicit_pullback::ImplicitPullback)((dy, dz))
    @unpack Aᵀ_op, Bᵀ_op, linear_solver, x = implicit_pullback
    R = eltype(x)
    dy_vec = convert(AbstractVector{R}, vec(unthunk(dy)))
    dF_vec = solve(linear_solver, Aᵀ_op, dy_vec)
    dx_vec = vec(similar(x))
    mul!(dx_vec, Bᵀ_op, dF_vec)
    lmul!(-one(R), dx_vec)
    dx = reshape(dx_vec, size(x))
    return (NoTangent(), dx, NoTangent())
end

end
