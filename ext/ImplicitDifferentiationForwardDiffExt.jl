module ImplicitDifferentiationForwardDiffExt

@static if isdefined(Base, :get_extension)
    using ForwardDiff: Dual, Partials, jacobian, partials, value
else
    using ..ForwardDiff: Dual, Partials, jacobian, partials, value
end

using AbstractDifferentiation: AbstractBackend, ForwardDiffBackend
using ImplicitDifferentiation:
    ImplicitDifferentiation,
    ImplicitFunction,
    DirectLinearSolver,
    IterativeLinearSolver,
    conditions_forward_operators,
    identity_break_autodiff,
    get_output,
    get_byproduct,
    presolve,
    solve
using LinearAlgebra: mul!
using PrecompileTools: @compile_workload

"""
    call_implicit_function(implicit, x_and_dx::AbstractArray{<:Dual}, args...; kwargs...)

Overload an [`ImplicitFunction`](@ref) on dual numbers to ensure compatibility with forward mode autodiff.

This is only available if ForwardDiff.jl is loaded (extension).

We compute the Jacobian-vector product `Jv` by solving `Au = -Bv` and setting `Jv = u`.
Positional and keyword arguments are passed to both `implicit.forward` and `implicit.conditions`.
"""
function ImplicitDifferentiation.call_implicit_function(
    implicit::ImplicitFunction, x_and_dx::AbstractArray{Dual{T,R,N}}, args...; kwargs...
) where {T,R,N}
    linear_solver = implicit.linear_solver

    x = value.(x_and_dx)
    y_or_yz = implicit(x, args...; kwargs...)
    y = get_output(y_or_yz)
    y_vec = vec(y)

    backend = forward_conditions_backend(implicit)
    A_vec, B_vec = conditions_forward_operators(backend, implicit, x, y_or_yz, args; kwargs)
    A_vec_presolved = presolve(linear_solver, A_vec, y)

    dy = ntuple(Val(N)) do k
        dₖx = partials.(x_and_dx, k)
        dₖx_vec = vec(dₖx)
        dₖc_vec = similar(y_vec)
        mul!(dₖc_vec, B_vec, dₖx_vec)
        dₖy_vec = solve(implicit.linear_solver, A_vec_presolved, -dₖc_vec)
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
