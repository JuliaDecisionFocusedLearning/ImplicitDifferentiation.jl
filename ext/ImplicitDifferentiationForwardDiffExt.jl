module ImplicitDifferentiationForwardDiffExt

@static if isdefined(Base, :get_extension)
    using ForwardDiff: Dual, Partials, jacobian, partials, value
else
    using ..ForwardDiff: Dual, Partials, jacobian, partials, value
end

using AbstractDifferentiation: AbstractBackend, ForwardDiffBackend
using ImplicitDifferentiation: ImplicitFunction, DirectLinearSolver, IterativeLinearSolver
using ImplicitDifferentiation: conditions_pushforwards, pushforward_to_operator
using ImplicitDifferentiation: get_output, get_byproduct, solve
using ImplicitDifferentiation: identity_break_autodiff
using LinearAlgebra: mul!
using PrecompileTools: @compile_workload

"""
    implicit(x_and_dx::AbstractArray{<:Dual}, args...; kwargs...)

Overload an [`ImplicitFunction`](@ref) on dual numbers to ensure compatibility with forward mode autodiff.

This is only available if ForwardDiff.jl is loaded (extension).

We compute the Jacobian-vector product `Jv` by solving `Au = -Bv` and setting `Jv = u`.
Positional and keyword arguments are passed to both `implicit.forward` and `implicit.conditions`.
"""
function (implicit::ImplicitFunction)(
    x_and_dx::AbstractArray{Dual{T,R,N}}, args...; kwargs...
) where {T,R,N}
    x = value.(x_and_dx)
    y_or_yz = implicit(x, args...; kwargs...)
    y = get_output(y_or_yz)

    backend = forward_conditions_backend(implicit)
    pfA, pfB = conditions_pushforwards(backend, implicit, x, y_or_yz, args; kwargs)
    A_vec = pushforward_to_operator(implicit, y, pfA)

    dy = ntuple(Val(N)) do k
        dₖx = partials.(x_and_dx, k)
        dₖc = pfB(dₖx)
        dₖc_vec = vec(dₖc)
        dₖy_vec = solve(implicit.linear_solver, A_vec, -dₖc_vec)
        reshape(dₖy_vec, size(y))
    end

    y_and_dy = map(eachindex(IndexCartesian(), y)) do i
        Dual{T}(y[i], Partials(ntuple(k -> dy[k][i], Val(N))))
    end

    if y_or_yz isa Tuple
        return y_and_dy, get_byproduct(y_or_yz)
    else
        return y_and_dy
    end
end

function forward_conditions_backend(::ImplicitFunction{F,C,L,Nothing}) where {F,C,L}
    return ForwardDiffBackend()
end

function forward_conditions_backend(
    implicit::ImplicitFunction{F,C,L,<:AbstractBackend}
) where {F,C,L}
    return implicit.conditions_backend
end

@compile_workload begin
    forward(x) = sqrt.(identity_break_autodiff(x))
    conditions(x, y) = y .^ 2 .- x
    for linear_solver in (DirectLinearSolver(), IterativeLinearSolver())
        implicit = ImplicitFunction(forward, conditions; linear_solver)
        x = rand(2)
        implicit(x)
        jacobian(implicit, x)
    end
end

end
