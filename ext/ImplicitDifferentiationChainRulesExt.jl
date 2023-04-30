module ImplicitDifferentiationChainRulesExt

using AbstractDifferentiation: ReverseRuleConfigBackend, pullback_function
using ChainRulesCore: ChainRulesCore, NoTangent, RuleConfig, ZeroTangent, unthunk
using ImplicitDifferentiation: ImplicitFunction, PullbackMul!, check_solution
using LinearOperators: LinearOperator

"""
    rrule(rc, implicit, x[; kwargs...])

Custom reverse rule for [`ImplicitFunction{F,C,L}`](@ref).

We compute the vector-Jacobian product `Jᵀv` by solving `Aᵀu = v` and setting `Jᵀv = -Bᵀu`.
Keyword arguments are given to both `implicit.forward` and `implicit.conditions`.
"""
function ChainRulesCore.rrule(
    rc::RuleConfig, implicit::ImplicitFunction, x::AbstractArray{R}; kwargs...
) where {R<:Real}
    conditions = implicit.conditions
    linear_solver = implicit.linear_solver

    y, z = implicit(x; kwargs...)
    n, m = length(x), length(y)

    backend = ReverseRuleConfigBackend(rc)
    pbA = pullback_function(backend, _y -> conditions(x, _y, z; kwargs...), y)
    pbB = pullback_function(backend, _x -> conditions(_x, y, z; kwargs...), x)
    Aᵀ_op = LinearOperator(R, m, m, false, false, PullbackMul!(pbA, size(y)))
    Bᵀ_op = LinearOperator(R, n, m, false, false, PullbackMul!(pbB, size(y)))

    function implicit_pullback((dy, dz))
        dy_vec = convert(Vector{R}, vec(unthunk(dy)))
        dF_vec, stats = linear_solver(Aᵀ_op, dy_vec)
        check_solution(linear_solver, stats)
        dx_vec = -(Bᵀ_op * dF_vec)
        dx = reshape(dx_vec, size(x))
        return (NoTangent(), dx)
    end

    return (y, z), implicit_pullback
end

end
