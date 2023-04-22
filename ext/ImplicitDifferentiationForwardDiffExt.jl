module ImplicitDifferentiationForwardDiffExt

@static if isdefined(Base, :get_extension)
    using ForwardDiff: Dual, Partials, jacobian, partials, value
else
    using ..ForwardDiff: Dual, Partials, jacobian, partials, value
end

using AbstractDifferentiation: ForwardDiffBackend, pushforward_function
using ImplicitDifferentiation: ImplicitFunction, SolverFailureException
using LinearOperators: LinearOperator

"""
    implicit(x_and_dx::AbstractArray{ForwardDiff.Dual}[; kwargs...])

Overload an [`ImplicitFunction`](@ref) on dual numbers to ensure compatibility with ForwardDiff.jl.
"""
function (implicit::ImplicitFunction)(
    x_and_dx::AbstractArray{Dual{T,R,N}}; kwargs...
) where {T,R,N}
    forward = implicit.forward
    conditions = implicit.conditions
    linear_solver = implicit.linear_solver

    x = value.(x_and_dx)
    y, z = forward(x)
    n, m = length(x), length(y)

    backend = ForwardDiffBackend()
    pushforward_A = pushforward_function(backend, _y -> conditions(x, _y, z), y)
    pushforward_B = pushforward_function(backend, _x -> conditions(_x, y, z), x)

    function mul_A!(res::Vector, dy_vec::Vector)
        dy = reshape(dy_vec, size(y))
        dF = only(pushforward_A(dy))
        return res .= vec(dF)
    end

    function mul_B!(res::Vector, dx_vec::Vector)
        dx = reshape(dx_vec, size(x))
        dF = only(pushforward_B(dx))
        return res .= vec(dF)
    end

    A = LinearOperator(R, m, m, false, false, mul_A!)
    B = LinearOperator(R, m, n, false, false, mul_B!)

    dy = map(1:N) do k
        dₖx_vec = vec(partials.(x_and_dx, k))
        dₖy_vec, stats = linear_solver(A, -B * dₖx_vec)
        if !stats.solved
            throw(SolverFailureException("Linear solver failed to converge", stats))
        end
        reshape(dₖy_vec, size(y))
    end

    y_and_dy = map(eachindex(y)) do i
        Dual{T}(y[i], Partials(Tuple(dy[k][i] for k in 1:N)))
    end

    return y_and_dy
end

end
