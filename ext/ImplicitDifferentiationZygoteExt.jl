module ImplicitDifferentiationZygoteExt

using ADTypes: AutoForwardDiff, AutoZygote
using ImplicitDifferentiation: ImplicitDifferentiation
using Zygote: ZygoteRuleConfig

ImplicitDifferentiation.suggested_forward_backend(::ZygoteRuleConfig) = AutoForwardDiff()
ImplicitDifferentiation.suggested_reverse_backend(::ZygoteRuleConfig) = AutoZygote()

end
