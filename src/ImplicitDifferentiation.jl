module ImplicitDifferentiation

using ChainRulesCore
using LinearOperators
using NamedTupleTools
using SparseArrays


# include("flatten/utils.jl")
# include("flatten/unflatten.jl")
# include("flatten/flatten.jl")
# include("flatten/flatten_similar.jl")
# include("flatten/unclear.jl")
include("flatten/flatten_nonconvexcore.jl")

include("implicit_function.jl")

include("simplex.jl")

export ImplicitFunction
export simplex_projection

end
