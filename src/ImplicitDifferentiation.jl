module ImplicitDifferentiation

using ChainRulesCore
using LinearOperators
using NamedTupleTools
using SparseArrays

# using ParameterHandling
# include("flatten.jl")

include("flatten_nonconvexcore.jl")
include("implicit_function.jl")
include("simplex.jl")

export ImplicitFunction
export simplex_projection

end
