module ImplicitDifferentiationForwardDiffExt

@static if isdefined(Base, :get_extension)
    using ForwardDiff: Dual, Partials, jacobian, partials, value
else
    using ..ForwardDiff: Dual, Partials, jacobian, partials, value
end

using AbstractDifferentiation: AbstractBackend, ForwardDiffBackend, pushforward_function
using ImplicitDifferentiation: ImplicitFunction, DirectLinearSolver, IterativeLinearSolver
using ImplicitDifferentiation: forward_operators, solve, identity_break_autodiff
using PrecompileTools: @compile_workload

"""
    implicit(x_and_dx::AbstractArray{<:Dual}; kwargs...)

Overload an [`ImplicitFunction`](@ref) on dual numbers to ensure compatibility with forward mode autodiff.

This is only available if ForwardDiff.jl is loaded (extension).

We compute the Jacobian-vector product `Jv` by solving `Au = Bv` and setting `Jv = u`.
Keyword arguments are given to both `implicit.forward` and `implicit.conditions`.
"""
function (implicit::ImplicitFunction)(
    x_and_dx::AbstractArray{Dual{T,R,N}}; kwargs...
) where {T,R,N}
    x = value.(x_and_dx)
    y_or_yz = implicit(x; kwargs...)
    y = _output(y_or_yz)

    backend = ForwardDiffBackend()
    A_op, B_op = forward_operators(backend, implicit, x, y_or_yz; kwargs)

    dy = ntuple(Val(N)) do k
        dₖx_vec = vec(partials.(x_and_dx, k))
        Bdx = B_op * dₖx_vec
        dₖy_vec = -solve(implicit.linear_solver, A_op, Bdx)
        reshape(dₖy_vec, size(y))
    end

    y_and_dy = let y = y, dy = dy
        y_and_dy_vec = map(eachindex(y)) do i
            Dual{T}(y[i], Partials(ntuple(k -> dy[k][i], Val(N))))
        end
        reshape(y_and_dy_vec, size(y))
    end

    if y_or_yz isa Tuple
        return y_and_dy, _byproduct(y_or_yz)
    else
        return y_and_dy
    end
end

_output(y::AbstractArray) = y
_output(yz::Tuple) = yz[1]
_byproduct(yz::Tuple) = yz[2]

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
