module ImplicitDifferentiation

using ChainRulesCore
using LinearOperators
using NamedTupleTools
using ParameterHandling

include("flatten.jl")
include("implicit_function.jl")
include("simplex.jl")

export ImplicitFunction

end
