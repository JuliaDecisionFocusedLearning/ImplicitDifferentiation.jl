module ImplicitDifferentiationChainRulesCoreExt

using AbstractDifferentiation: AbstractBackend, ReverseRuleConfigBackend
using ChainRulesCore: ChainRulesCore, NoTangent, RuleConfig
using ChainRulesCore: rrule, unthunk, @not_implemented
using ImplicitDifferentiation: ImplicitFunction, reverse_operators, solve
using LinearAlgebra: lmul!, mul!
using SimpleUnPack: @unpack

"""
    rrule(rc, implicit, x, args...; kwargs...)

Custom reverse rule for an [`ImplicitFunction`](@ref), to ensure compatibility with reverse mode autodiff.

This is only available if ChainRulesCore.jl is loaded (extension), except on Julia < 1.9 where it is always available.

We compute the vector-Jacobian product `Jᵀv` by solving `Aᵀu = v` and setting `Jᵀv = -Bᵀu`.
Positional and keyword arguments are passed to both `implicit.forward` and `implicit.conditions`.
"""
function ChainRulesCore.rrule(
    rc::RuleConfig, implicit::ImplicitFunction, x::AbstractArray{R}, args...; kwargs...
) where {R}
    y_or_yz = implicit(x, args...; kwargs...)
    backend = reverse_conditions_backend(rc, implicit)
    Aᵀ_vec, pbBᵀ = reverse_operators(backend, implicit, x, y_or_yz, args; kwargs)
    byproduct = y_or_yz isa Tuple
    nbargs = length(args)
    implicit_pullback = ImplicitPullback{byproduct,nbargs}(
        Aᵀ_vec, pbBᵀ, implicit.linear_solver
    )
    return y_or_yz, implicit_pullback
end

function reverse_conditions_backend(
    rc::RuleConfig, ::ImplicitFunction{F,C,L,Nothing}
) where {F,C,L}
    return ReverseRuleConfigBackend(rc)
end

function reverse_conditions_backend(
    ::RuleConfig, implicit::ImplicitFunction{F,C,L,<:AbstractBackend}
) where {F,C,L}
    return implicit.conditions_backend
end

struct ImplicitPullback{byproduct,nbargs,A,B,L}
    Aᵀ_vec::A
    pbBᵀ::B
    linear_solver::L

    function ImplicitPullback{byproduct,nbargs}(
        Aᵀ_vec::A, pbBᵀ::B, linear_solver::L
    ) where {byproduct,nbargs,A,B,L}
        return new{byproduct,nbargs,A,B,L}(Aᵀ_vec, pbBᵀ, linear_solver)
    end
end

function (implicit_pullback::ImplicitPullback{true})((dy, dz))
    return _apply(implicit_pullback, dy)
end

function (implicit_pullback::ImplicitPullback{false})(dy)
    return _apply(implicit_pullback, dy)
end

function unimplemented_tangent(_)
    return @not_implemented(
        "Tangents for positional arguments of an ImplicitFunction beyond x (the first one) are not implemented"
    )
end

function _apply(
    implicit_pullback::ImplicitPullback{byproduct,nbargs}, dy_thunk
) where {byproduct,nbargs}
    @unpack Aᵀ_vec, pbBᵀ, linear_solver = implicit_pullback
    dy = unthunk(dy_thunk)
    dy_vec = vec(dy)
    dc_vec = solve(linear_solver, Aᵀ_vec, dy_vec)
    dc = reshape(dc_vec, size(dy))
    dx = -pbBᵀ(dc)
    return (NoTangent(), dx, ntuple(unimplemented_tangent, nbargs)...)
end

end
