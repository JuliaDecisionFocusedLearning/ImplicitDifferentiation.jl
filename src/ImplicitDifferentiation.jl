module ImplicitDifferentiation

using ChainRulesCore: ChainRulesCore, NoTangent, RuleConfig
using ChainRulesCore: frule_via_ad, rrule_via_ad, unthunk
using Krylov: gmres
using LinearOperators: LinearOperator
using SparseArrays

include("implicit_function.jl")
include("simplex.jl")

export ImplicitFunction
export simplex_projection

end
