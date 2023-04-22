module ImplicitDifferentiation

using AbstractDifferentiation: AbstractDifferentiation
using ChainRulesCore: ChainRulesCore, HasForwardsMode, NoTangent, RuleConfig, ZeroTangent
using ChainRulesCore: frule_via_ad, rrule_via_ad, unthunk
using Krylov: gmres
using LinearOperators: LinearOperator
using Requires: @require

include("implicit_function.jl")

export ImplicitFunction

@static if !isdefined(Base, :get_extension)
    function __init__()
        @require Enzyme = "7da242da-08ed-463a-9acd-ee780be4f1d9" include(
            "../ext/ImplicitDifferentiationEnzymeExt.jl"
        )
        @require ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210" include(
            "../ext/ImplicitDifferentiationForwardDiffExt.jl"
        )
    end
end

end
