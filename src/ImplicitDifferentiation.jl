module ImplicitDifferentiation

using ChainRulesCore
using LinearOperators
using NamedTupleTools
using SparseArrays

include("flatten.jl")
include("implicit_function.jl")
include("simplex.jl")

export ImplicitFunction
export simplex_projection

end