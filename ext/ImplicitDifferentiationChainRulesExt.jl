module ImplicitDifferentiationChainRulesExt

using AbstractDifferentiation: ReverseRuleConfigBackend, lazy_jacobian
using ChainRulesCore: ChainRulesCore, NoTangent, RuleConfig, ZeroTangent, unthunk
using ImplicitDifferentiation:
    ImplicitFunction, SolverFailureException, LazyJacobianTransposeMul!
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
    A = lazy_jacobian(backend, _y -> conditions(x, _y, z; kwargs...), y)
    B = lazy_jacobian(backend, _x -> conditions(_x, y, z; kwargs...), x)
    Aᵀ_op = LinearOperator(R, m, m, false, false, LazyJacobianTransposeMul!(A, size(y)))
    Bᵀ_op = LinearOperator(R, n, m, false, false, LazyJacobianTransposeMul!(B, size(y)))

    function implicit_pullback((dy, dz))
        dy_vec = convert(Vector{R}, vec(unthunk(dy)))
        dF_vec, stats = linear_solver(Aᵀ_op, dy_vec)
        if !stats.solved
            throw(SolverFailureException("Linear solver failed to converge", stats))
        end
        dx_vec = -(Bᵀ_op * dF_vec)
        dx = reshape(dx_vec, size(x))
        return (NoTangent(), dx)
    end

    return (y, z), implicit_pullback
end

end
