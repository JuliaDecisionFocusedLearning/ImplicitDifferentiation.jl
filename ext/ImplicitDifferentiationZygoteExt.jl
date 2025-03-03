module ImplicitDifferentiationZygoteExt

using ADTypes: AutoZygote
using ImplicitDifferentiation: ImplicitDifferentiation
using Zygote: ZygoteRuleConfig

ImplicitDifferentiation.chainrules_suggested_backend(::ZygoteRuleConfig) = AutoZygote()

end
