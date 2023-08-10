module ImplicitDifferentiationChainRulesCoreExt

using AbstractDifferentiation: AbstractBackend, ReverseRuleConfigBackend, ruleconfig
using ChainRulesCore: ChainRulesCore, NoTangent, RuleConfig
using ChainRulesCore: rrule, rrule_via_ad, unthunk, @not_implemented
using ImplicitDifferentiation: ImplicitDifferentiation
using ImplicitDifferentiation: ImplicitFunction
using ImplicitDifferentiation: conditions_pullbacks, pullbacks_to_operators
using ImplicitDifferentiation: get_output, solve
using LinearAlgebra: mul!
using SimpleUnPack: @unpack

"""
    rrule(rc, implicit, x, args...; kwargs...)

Custom reverse rule for an [`ImplicitFunction`](@ref), to ensure compatibility with reverse mode autodiff.

This is only available if ChainRulesCore.jl is loaded (extension), except on Julia < 1.9 where it is always available.

We compute the vector-Jacobian product `Jᵀv` by solving `Aᵀu = v` and setting `Jᵀv = -Bᵀu`.
Positional and keyword arguments are passed to both `implicit.forward` and `implicit.conditions`.
"""
function ChainRulesCore.rrule(
    rc::RuleConfig, implicit::ImplicitFunction, x::X, args...; kwargs...
) where {R,X<:AbstractArray{R}}
    y_or_yz = implicit(x, args...; kwargs...)
    y = get_output(y_or_yz)
    backend = reverse_conditions_backend(rc, implicit)
    pbAᵀ, pbBᵀ = conditions_pullbacks(backend, implicit, x, y_or_yz, args; kwargs)
    Aᵀ_vec, pbBᵀ = pullbacks_to_operators(implicit, x, y, pbAᵀ, pbBᵀ)

    byproduct = y_or_yz isa Tuple
    nbargs = length(args)
    implicit_pullback = ImplicitPullback{byproduct,nbargs}(
        Aᵀ_vec, pbBᵀ, implicit.linear_solver, vec(x), size(x)
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

struct ImplicitPullback{byproduct,nbargs,A,B,L,X,N}
    Aᵀ_vec::A
    pbBᵀ::B
    linear_solver::L
    x_vec::X
    x_size::NTuple{N,Int}

    function ImplicitPullback{byproduct,nbargs}(
        Aᵀ_vec::A, pbBᵀ::B, linear_solver::L, x_vec::X, x_size::NTuple{N,Int}
    ) where {byproduct,nbargs,A,B,L,X,N}
        return new{byproduct,nbargs,A,B,L,X,N}(Aᵀ_vec, pbBᵀ, linear_solver, x_vec, x_size)
    end
end

function (implicit_pullback::ImplicitPullback{true})((dy, dz))
    return apply_implicit_pullback(implicit_pullback, dy)
end

function (implicit_pullback::ImplicitPullback{false})(dy)
    return apply_implicit_pullback(implicit_pullback, dy)
end

function unimplemented_tangent(_)
    return @not_implemented(
        "Tangents for positional arguments of an ImplicitFunction beyond x (the first one) are not implemented"
    )
end

function apply_implicit_pullback(
    implicit_pullback::ImplicitPullback{byproduct,nbargs}, dy_thunk
) where {byproduct,nbargs}
    @unpack Aᵀ_vec, pbBᵀ, linear_solver, x_vec, x_size = implicit_pullback
    dy = unthunk(dy_thunk)
    dy_vec = vec(dy)
    dc_vec = solve(linear_solver, Aᵀ_vec, -dy_vec)
    dc = reshape(dc_vec, x_size)
    dx = pbBᵀ(dc)
    return (NoTangent(), dx, ntuple(unimplemented_tangent, nbargs)...)
end

end
