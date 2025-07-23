module ImplicitDifferentiationChainRulesCoreExt

using ADTypes: AutoChainRules
using ChainRulesCore: ChainRulesCore, NoTangent, ProjectTo, RuleConfig
using ChainRulesCore: unthunk, @not_implemented
using ImplicitDifferentiation:
    ImplicitDifferentiation,
    ImplicitFunction,
    ImplicitFunctionPreparation,
    build_Aᵀ,
    build_Bᵀ,
    chainrules_suggested_backend

# not covered by Codecov for now
ImplicitDifferentiation.chainrules_suggested_backend(rc::RuleConfig) = AutoChainRules(rc)

struct ImplicitPullback{TA,TB,TL,TC,TP,Nargs}
    Aᵀ::TA
    Bᵀ::TB
    linear_solver::TL
    c0::TC
    project_x::TP
    _Nargs::Val{Nargs}
end

function (pb::ImplicitPullback{TA,TB,TL,TC,TP,Nargs})((dy, dz)) where {TA,TB,TL,TP,TC,Nargs}
    (; Aᵀ, Bᵀ, linear_solver, c0, project_x) = pb
    dc = linear_solver(Aᵀ, -unthunk(dy), c0)
    dx = Bᵀ(dc)
    df = NoTangent()
    dargs = ntuple(unimplemented_tangent, Val(Nargs))
    return (df, project_x(dx), dargs...)
end

function ChainRulesCore.rrule(
    rc::RuleConfig, implicit::ImplicitFunction, x::AbstractArray, args::Vararg{Any,N};
) where {N}
    (; conditions, linear_solver) = implicit
    y, z = implicit(x, args...)
    c = conditions(x, y, z, args...)
    c0 = zero(c)

    suggested_backend = chainrules_suggested_backend(rc)
    prep = ImplicitFunctionPreparation(eltype(x))
    Aᵀ = build_Aᵀ(implicit, prep, x, y, z, c, args...; suggested_backend)
    Bᵀ = build_Bᵀ(implicit, prep, x, y, z, c, args...; suggested_backend)
    project_x = ProjectTo(x)

    implicit_pullback = ImplicitPullback(Aᵀ, Bᵀ, linear_solver, c0, project_x, Val(N))
    return (y, z), implicit_pullback
end

function unimplemented_tangent(_)
    return @not_implemented(
        "Tangents for positional arguments of an `ImplicitFunction` beyond `x` (the first one) are not implemented"
    )
end
end
