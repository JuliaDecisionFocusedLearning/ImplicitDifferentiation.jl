module ImplicitDifferentiationChainRulesCoreExt

using ADTypes: AbstractADType, AutoChainRules
using ChainRulesCore: ChainRulesCore, NoTangent, ProjectTo, RuleConfig
using ChainRulesCore: rrule, rrule_via_ad, unthunk, @not_implemented
using ImplicitDifferentiation: ImplicitFunction, build_Aᵀ, build_Bᵀ, output

function ChainRulesCore.rrule(
    rc::RuleConfig,
    implicit::ImplicitFunction,
    x::AbstractVector,
    args::Vararg{T,N};
    kwargs...,
) where {T,N}
    y_or_yz = implicit(x, args...; kwargs...)

    suggested_backend = AutoChainRules(rc)
    Aᵀ = build_Aᵀ(implicit, x, y_or_yz, args...; suggested_backend, kwargs...)
    Bᵀ = build_Bᵀ(implicit, x, y_or_yz, args...; suggested_backend, kwargs...)
    project_x = ProjectTo(x)

    implicit_pullback = ImplicitPullback(
        Aᵀ, Bᵀ, implicit.linear_solver, project_x, Val{N}()
    )
    return y_or_yz, implicit_pullback
end

struct ImplicitPullback{N,M1,M2,L,P}
    Aᵀ::M1
    Bᵀ::M2
    linear_solver::L
    project_x::P
    nargs::Val{N}
end

function (ip::ImplicitPullback{N})(dy_or_dydz) where {N}
    (; Aᵀ, Bᵀ, linear_solver, project_x) = ip
    dy = output(unthunk(dy_or_dydz))
    dc = linear_solver(Aᵀ, -dy)
    dx = Bᵀ * dc
    df = NoTangent()
    dargs = ntuple(unimplemented_tangent, N)
    return (df, project_x(dx), dargs...)
end

function unimplemented_tangent(_)
    return @not_implemented(
        "Tangents for positional arguments of an `ImplicitFunction` beyond `x` (the first one) are not implemented"
    )
end

end
