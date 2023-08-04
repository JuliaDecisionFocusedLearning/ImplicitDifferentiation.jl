module ImplicitDifferentiationChainRulesExt

using AbstractDifferentiation: ReverseRuleConfigBackend
using ChainRulesCore: ChainRulesCore, NoTangent, RuleConfig, ZeroTangent, rrule, unthunk
using ImplicitDifferentiation: ImplicitFunction, reverse_operators, solve
using SimpleUnPack: @unpack

"""
    rrule(rc, implicit, x; kwargs...)

Custom reverse rule for an [`ImplicitFunction`](@ref), to ensure compatibility with reverse mode autodiff.

This is only available if ChainRulesCore.jl is loaded (extension), except on Julia < 1.9 where it is always available.

We compute the vector-Jacobian product `Jᵀv` by solving `Aᵀu = v` and setting `Jᵀv = -Bᵀu`.
Keyword arguments are given to both `implicit.forward` and `implicit.conditions`.
"""
function ChainRulesCore.rrule(
    rc::RuleConfig, implicit::ImplicitFunction, x::AbstractArray{R}; kwargs...
) where {R}
    y_or_yz = implicit(x; kwargs...)
    backend = ReverseRuleConfigBackend(rc)
    Aᵀ_op, Bᵀ_op = reverse_operators(backend, implicit, x, y_or_yz; kwargs)
    byproduct = y_or_yz isa Tuple
    implicit_pullback = ImplicitPullback{byproduct}(Aᵀ_op, Bᵀ_op, implicit.linear_solver, x)
    return y_or_yz, implicit_pullback
end

struct ImplicitPullback{byproduct,A,B,L,X}
    Aᵀ_op::A
    Bᵀ_op::B
    linear_solver::L
    x::X

    function ImplicitPullback{byproduct}(
        Aᵀ_op::A, Bᵀ_op::B, linear_solver::L, x::X
    ) where {byproduct,A,B,L,X}
        return new{byproduct,A,B,L,X}(Aᵀ_op, Bᵀ_op, linear_solver, x)
    end
end

function (implicit_pullback::ImplicitPullback{true})((dy, dz))
    return _apply(implicit_pullback, dy)
end

function (implicit_pullback::ImplicitPullback{false})(dy)
    return _apply(implicit_pullback, dy)
end

function _apply(implicit_pullback::ImplicitPullback, dy)
    @unpack Aᵀ_op, Bᵀ_op, linear_solver, x = implicit_pullback
    R = eltype(x)
    dy_vec = convert(AbstractVector{R}, vec(unthunk(dy)))
    dF_vec = solve(linear_solver, Aᵀ_op, dy_vec)
    dx_vec = Bᵀ_op * (-dF_vec)
    dx = reshape(dx_vec, size(x))
    return (NoTangent(), dx, NoTangent())
end

end
