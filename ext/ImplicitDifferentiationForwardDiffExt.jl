module ImplicitDifferentiationForwardDiffExt

@static if isdefined(Base, :get_extension)
    using ForwardDiff: Dual, Partials, jacobian, partials, value
else
    using ..ForwardDiff: Dual, Partials, jacobian, partials, value
end

using AbstractDifferentiation: ForwardDiffBackend, pushforward_function
using ImplicitDifferentiation: ImplicitFunction, PushforwardMul!, check_solution
using LinearOperators: LinearOperator

"""
    implicit(x_and_dx::AbstractArray{ForwardDiff.Dual}[; kwargs...])

Overload an [`ImplicitFunction`](@ref) on dual numbers to ensure compatibility with ForwardDiff.jl.
"""
function (implicit::ImplicitFunction)(
    x_and_dx::AbstractArray{Dual{T,R,N}}; kwargs...
) where {T,R,N}
    conditions = implicit.conditions
    linear_solver = implicit.linear_solver

    x = value.(x_and_dx)
    y, z = implicit(x; kwargs...)
    n, m = length(x), length(y)

    backend = ForwardDiffBackend()
    pfA = pushforward_function(backend, _y -> conditions(x, _y, z; kwargs...), y)
    pfB = pushforward_function(backend, _x -> conditions(_x, y, z; kwargs...), x)
    A_op = LinearOperator(R, m, m, false, false, PushforwardMul!(pfA, size(y)))
    B_op = LinearOperator(R, m, n, false, false, PushforwardMul!(pfB, size(x)))

    dy = map(1:N) do k
        dₖx_vec = vec(partials.(x_and_dx, k))
        dₖy_vec, stats = linear_solver(A_op, -(B_op * dₖx_vec))
        check_solution(linear_solver, stats)
        reshape(dₖy_vec, size(y))
    end

    y_and_dy = let y = y, dy = dy
        map(eachindex(y)) do i
            Dual{T}(y[i], Partials(ntuple(k -> dy[k][i], Val(N))))
        end
    end

    z_and_dz = let z = z
        Dual{T}(z, Partials(ntuple(_ -> zero(z), Val(N))))
    end

    return y_and_dy, z_and_dz
end

end