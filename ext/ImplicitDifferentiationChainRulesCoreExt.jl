module ImplicitDifferentiationChainRulesCoreExt

using ADTypes: AutoChainRules, AutoForwardDiff
using ChainRulesCore: ChainRulesCore, NoTangent, ProjectTo, RuleConfig
using ChainRulesCore: unthunk, @not_implemented
using ImplicitDifferentiation:
    ImplicitDifferentiation,
    ImplicitFunction,
    ImplicitFunctionPreparation,
    IterativeLeastSquaresSolver,
    build_Aᵀ,
    build_Bᵀ,
    suggested_forward_backend,
    suggested_reverse_backend

# not covered by Codecov for now
ImplicitDifferentiation.suggested_forward_backend(rc::RuleConfig) = AutoForwardDiff()
ImplicitDifferentiation.suggested_reverse_backend(rc::RuleConfig) = AutoChainRules(rc)

struct ImplicitPullback{Nargs,TA,TB,TA2,TL,TC,TP}
    Aᵀ::TA
    Bᵀ::TB
    A::TA2
    linear_solver::TL
    c0::TC
    project_x::TP
    _Nargs::Val{Nargs}
end

function (pb::ImplicitPullback{Nargs})((dy, dz)) where {Nargs}
    (; Aᵀ, Bᵀ, A, linear_solver, c0, project_x) = pb
    dc = linear_solver(Aᵀ, A, -unthunk(dy), c0)
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

    forward_backend = suggested_forward_backend(rc)
    reverse_backend = suggested_reverse_backend(rc)
    prep = ImplicitFunctionPreparation(eltype(x))
    Aᵀ = build_Aᵀ(implicit, prep, x, y, z, c, args...; suggested_backend=reverse_backend)
    Bᵀ = build_Bᵀ(implicit, prep, x, y, z, c, args...; suggested_backend=reverse_backend)
    if linear_solver isa IterativeLeastSquaresSolver
        A = build_Aᵀ(implicit, prep, x, y, z, c, args...; suggested_backend=forward_backend)
    else
        A = nothing
    end
    project_x = ProjectTo(x)

    implicit_pullback = ImplicitPullback(Aᵀ, Bᵀ, A, linear_solver, c0, project_x, Val(N))
    return (y, z), implicit_pullback
end

function unimplemented_tangent(_)
    return @not_implemented(
        "Tangents for positional arguments of an `ImplicitFunction` beyond `x` (the first one) are not implemented"
    )
end
end
