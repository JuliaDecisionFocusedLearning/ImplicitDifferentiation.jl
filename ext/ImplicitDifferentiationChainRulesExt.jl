module ImplicitDifferentiationChainRulesExt

using AbstractDifferentiation: ReverseRuleConfigBackend, pullback_function
using ChainRulesCore: ChainRulesCore, NoTangent, RuleConfig, ZeroTangent, unthunk
using ImplicitDifferentiation: ImplicitFunction, PullbackMul!, check_solution
using LinearOperators: LinearOperator
using SimpleUnPack: @unpack
using LinearAlgebra: mul!

"""
    rrule(rc, implicit, x; kwargs...)
    rrule(rc, implicit, x, Val(return_byproduct); kwargs...)

Custom reverse rule for an [`ImplicitFunction`](@ref), to ensure compatibility with reverse mode autodiff.

This is only available if ChainRulesCore.jl is loaded (extension).

- If `return_byproduct=false` (the default), this returns a single output `y(x)` with a pullback accepting a single cotangent `̄y`.
- If `return_byproduct=true`, this returns a couple of outputs `(y(x),z(x))` with a pullback accepting a couple of cotangents `(̄y, ̄z)` (remember that `z(x)` is not differentiated so its cotangent is ignored).

We compute the vector-Jacobian product `Jᵀv` by solving `Aᵀu = v` and setting `Jᵀv = -Bᵀu`.
Keyword arguments are given to both `implicit.forward` and `implicit.conditions`.
"""
function ChainRulesCore.rrule(
    rc::RuleConfig,
    implicit::ImplicitFunction,
    x::AbstractArray{R},
    ::Val{return_byproduct};
    kwargs...,
) where {R,return_byproduct}
    @unpack conditions, linear_solver, presolver = implicit

    y, z = implicit(x, Val(true); kwargs...)
    n, m = length(x), length(y)

    backend = ReverseRuleConfigBackend(rc)
    pbA = pullback_function(backend, _y -> conditions(x, _y, z; kwargs...), y)
    pbB = pullback_function(backend, _x -> conditions(_x, y, z; kwargs...), x)
    pbmA = PullbackMul!(pbA, size(y))
    pbmB = PullbackMul!(pbB, size(y))
    Aᵀ_op = LinearOperator(R, m, m, false, false, pbmA)
    Bᵀ_op = LinearOperator(R, n, m, false, false, pbmB)
    implicit_pullback = ImplicitPullback(
        presolver(Aᵀ_op, x, y), Bᵀ_op, linear_solver, x, Val(return_byproduct)
    )

    if return_byproduct
        return (y, z), implicit_pullback
    else
        return y, implicit_pullback
    end
end

struct ImplicitPullback{return_byproduct,A,B,L,X}
    Aᵀ_op::A
    Bᵀ_op::B
    linear_solver::L
    x::X
    _v::Val{return_byproduct}
end

function (implicit_pullback_nobyproduct::ImplicitPullback{false})(dy)
    @unpack Aᵀ_op, Bᵀ_op, linear_solver, x = implicit_pullback_nobyproduct
    implicit_pullback_byproduct = ImplicitPullback(
        Aᵀ_op, Bᵀ_op, linear_solver, x, Val(true)
    )
    return implicit_pullback_byproduct((dy, nothing))
end

function (implicit_pullback_byproduct::ImplicitPullback{true})((dy, _))
    @unpack Aᵀ_op, Bᵀ_op, linear_solver, x = implicit_pullback_byproduct
    R = eltype(x)
    dy_vec = convert(AbstractVector{R}, vec(unthunk(dy)))
    dF_vec, stats = linear_solver(Aᵀ_op, dy_vec)
    check_solution(linear_solver, stats)
    dx_vec = vec(similar(x))
    mul!(dx_vec, Bᵀ_op, dF_vec)
    dx_vec .*= -1
    dx = reshape(dx_vec, size(x))
    return (NoTangent(), dx, NoTangent())
end

end
