module ImplicitDifferentiationChainRulesCoreExt

using ADTypes: AbstractADType, AutoChainRules
using ChainRulesCore: ChainRulesCore, NoTangent, ProjectTo, RuleConfig
using ChainRulesCore: rrule, rrule_via_ad, unthunk, @not_implemented
using ImplicitDifferentiation: ImplicitFunction, build_Aᵀ, build_Bᵀ, get_output
using LinearAlgebra: mul!

function ChainRulesCore.rrule(
    rc::RuleConfig, implicit::ImplicitFunction, x::AbstractVector, args...; kwargs...
)
    y_or_yz = implicit(x, args...; kwargs...)

    suggested_backend = AutoChainRules(rc)
    Aᵀ = build_Aᵀ(implicit, x, y_or_yz, args...; suggested_backend, kwargs...)
    Bᵀ = build_Bᵀ(implicit, x, y_or_yz, args...; suggested_backend, kwargs...)
    project_x = ProjectTo(x)

    function implicit_pullback(dy_or_dydz)
        dy = get_output(unthunk(dy_or_dydz))
        dc = implicit.linear_solver(Aᵀ, -dy)
        dx = Bᵀ * dc
        return (NoTangent(), project_x(dx), ntuple(unimplemented_tangent, length(args))...)
    end

    return y_or_yz, implicit_pullback
end

function unimplemented_tangent(_)
    return @not_implemented(
        "Tangents for positional arguments of an `ImplicitFunction` beyond `x` (the first one) are not implemented"
    )
end

end
