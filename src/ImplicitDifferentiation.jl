module ImplicitDifferentiation

using ChainRulesCore
using Krylov
using LinearOperators
using SparseArrays

include("implicit_function.jl")
include("simplex.jl")

export ImplicitFunction
export simplex_projection

end
