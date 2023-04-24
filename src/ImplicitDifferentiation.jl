module ImplicitDifferentiation

using AbstractDifferentiation: LazyJacobian, ReverseRuleConfigBackend, lazy_jacobian
using Krylov: gmres
using LinearOperators: LinearOperator
using Requires: @require

include("utils.jl")
include("implicit_function.jl")

export ImplicitFunction

@static if !isdefined(Base, :get_extension)
    include("../ext/ImplicitDifferentiationChainRulesExt.jl")
    function __init__()
        @require ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210" begin
            include("../ext/ImplicitDifferentiationForwardDiffExt.jl")
        end
    end
end

end
