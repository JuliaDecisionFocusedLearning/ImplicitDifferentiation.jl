module ImplicitDifferentiationChainRulesCoreExt

using ADTypes: AbstractADType, AutoChainRules
using ChainRulesCore: ChainRulesCore, NoTangent, ProjectTo, RuleConfig
using ChainRulesCore: rrule, rrule_via_ad, unthunk, @not_implemented
using ImplicitDifferentiation: ImplicitFunction, build_Aᵀ, build_Bᵀ

function ChainRulesCore.rrule(
    rc::RuleConfig,
    implicit::ImplicitFunction,
    x::AbstractVector,
    args::Vararg{Any,N};
    kwargs...,
) where {N}
    y, z = implicit(x, args...; kwargs...)

    Aᵀ = build_Aᵀ(implicit, x, y, z, args...)
    Bᵀ = build_Bᵀ(implicit, x, y, z, args...)
    project_x = ProjectTo(x)

    function implicit_pullback((dy, dz))
        dy = unthunk(dy)
        dc = implicit.linear_solver(Aᵀ, -dy)
        dx = Bᵀ * dc
        df = NoTangent()
        dargs = ntuple(unimplemented_tangent, N)
        return (df, project_x(dx), dargs...)
    end

    return (y, z), implicit_pullback
end

function unimplemented_tangent(_)
    return @not_implemented(
        "Tangents for positional arguments of an `ImplicitFunction` beyond `x` (the first one) are not implemented"
    )
end

end
