module ImplicitDifferentiationForwardDiffExt

@static if isdefined(Base, :get_extension)
    using ForwardDiff: Dual, Partials, jacobian, partials, value
else
    using ..ForwardDiff: Dual, Partials, jacobian, partials, value
end

using AbstractDifferentiation: ForwardDiffBackend, pushforward_function
using ImplicitDifferentiation: ImplicitFunction, PushforwardMul!, check_solution
using LinearOperators: LinearOperator
using SimpleUnPack: @unpack

"""
    implicit(x_and_dx::AbstractArray{<:Dual}; kwargs...)
    implicit(x_and_dx::AbstractArray{<:Dual}, Val(return_byproduct); kwargs...)

Overload an [`ImplicitFunction`](@ref) on dual numbers to ensure compatibility with ForwardDiff.jl.

- If `return_byproduct=false` (the default), this returns a single output `y_and_dy(x)`.
- If `return_byproduct=true`, this returns a couple of outputs `(y_and_dy(x),z(x))` (remember that `z(x)` is not differentiated so `dz(x)` doesn't exist).

We compute the Jacobian-vector product `Jv` by solving `Au = Bv` and setting `Jv = u`.
Keyword arguments are given to both `implicit.forward` and `implicit.conditions`.
"""
function (implicit::ImplicitFunction)(
    x_and_dx::AbstractArray{Dual{T,R,N}}, ::Val{return_byproduct}=Val(false); kwargs...
) where {T,R,N,return_byproduct}
    @unpack conditions, linear_solver = implicit

    x = value.(x_and_dx)
    y, z = implicit(x, Val(true); kwargs...)
    n, m = length(x), length(y)

    backend = ForwardDiffBackend()
    pfA = pushforward_function(backend, _y -> conditions(x, _y, z; kwargs...), y)
    pfB = pushforward_function(backend, _x -> conditions(_x, y, z; kwargs...), x)
    A_op = LinearOperator(R, m, m, false, false, PushforwardMul!(pfA, size(y)))
    B_op = LinearOperator(R, m, n, false, false, PushforwardMul!(pfB, size(x)))

    dy = map(1:N) do k
        dₖx_vec = vec(partials.(x_and_dx, k))
        dₖy_vec, stats = linear_solver(A_op, B_op * dₖx_vec)
        dₖy_vec .*= -1
        check_solution(linear_solver, stats)
        reshape(dₖy_vec, size(y))
    end

    y_and_dy = let y = y, dy = dy
        y_and_dy_vec = map(eachindex(y)) do i
            Dual{T}(y[i], Partials(ntuple(k -> dy[k][i], Val(N))))
        end
        reshape(y_and_dy_vec, size(y))
    end

    if return_byproduct
        return y_and_dy, z
    else
        return y_and_dy
    end
end

end
