module ImplicitDifferentiationChainRulesExt

using AbstractDifferentiation: ReverseRuleConfigBackend, pullback_function
using ChainRulesCore: ChainRulesCore, NoTangent, RuleConfig, ZeroTangent, unthunk
using ImplicitDifferentiation: ImplicitFunction, PullbackMul!, check_solution
using LinearOperators: LinearOperator

"""
    rrule(rc, implicit, x[; kwargs...])

Custom reverse rule for [`ImplicitFunction{F,C,L}`](@ref).

We compute the vector-Jacobian product `Jᵀv` by solving `Aᵀu = v` and setting `Jᵀv = -Bᵀu`.
Keyword arguments are given to both `implicit.forward` and `implicit.conditions`.
"""
function ChainRulesCore.rrule(
    rc::RuleConfig,
    implicit::ImplicitFunction,
    x::AbstractArray{R},
    ::Val{byproduct};
    kwargs...,
) where {R,byproduct}
    conditions = implicit.conditions
    linear_solver = implicit.linear_solver

    y, z = implicit(x, Val(true); kwargs...)
    n, m = length(x), length(y)

    backend = ReverseRuleConfigBackend(rc)
    pbA = pullback_function(backend, _y -> conditions(x, _y, z; kwargs...), y)
    pbB = pullback_function(backend, _x -> conditions(_x, y, z; kwargs...), x)
    pbmA = PullbackMul!(pbA, size(y))
    pbmB = PullbackMul!(pbB, size(y))
    Aᵀ_op = LinearOperator(R, m, m, false, false, pbmA)
    Bᵀ_op = LinearOperator(R, n, m, false, false, pbmB)
    implicit_pullback = ImplicitPullback(Aᵀ_op, Bᵀ_op, linear_solver, x, Val(byproduct))

    return (byproduct ? (y, z) : y), implicit_pullback
end

struct ImplicitPullback{byproduct,A,B,L,X}
    Aᵀ_op::A
    Bᵀ_op::B
    linear_solver::L
    x::X
    _v::Val{byproduct}
end

function (pb::ImplicitPullback{false})(dy)
    _pb = ImplicitPullback(pb.Aᵀ_op, pb.Bᵀ_op, pb.linear_solver, pb.x, Val(true))
    return _pb((dy, nothing))
end
function (implicit_pullback::ImplicitPullback{true})((dy, _))
    Aᵀ_op = implicit_pullback.Aᵀ_op
    Bᵀ_op = implicit_pullback.Bᵀ_op
    linear_solver = implicit_pullback.linear_solver
    x = implicit_pullback.x
    R = eltype(x)

    dy_vec = convert(Vector{R}, vec(unthunk(dy)))
    dF_vec, stats = linear_solver(Aᵀ_op, dy_vec)
    check_solution(linear_solver, stats)
    dx_vec = Bᵀ_op * dF_vec
    dx_vec .*= -1
    dx = reshape(dx_vec, size(x))
    return (NoTangent(), dx, NoTangent())
end

end
