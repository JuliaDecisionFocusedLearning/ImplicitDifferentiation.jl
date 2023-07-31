module ImplicitDifferentiationForwardDiffExt

@static if isdefined(Base, :get_extension)
    using ForwardDiff: Dual, Partials, jacobian, partials, value
else
    using ..ForwardDiff: Dual, Partials, jacobian, partials, value
end

using AbstractDifferentiation:
    AbstractDifferentiation, ForwardDiffBackend, pushforward_function
using ImplicitDifferentiation:
    ImplicitFunction, PushforwardMul!, ReturnByproduct, presolve, solve
using LinearAlgebra: lmul!, mul!
using LinearOperators: LinearOperator
using SimpleUnPack: @unpack

"""
    implicit(x_and_dx::AbstractArray{<:Dual}[, ReturnByproduct()]; kwargs...)

Overload an [`ImplicitFunction`](@ref) on dual numbers to ensure compatibility with forward mode autodiff.

This is only available if ForwardDiff.jl is loaded (extension).

- By default, this returns a single output `y_and_dy(x)`.
- If `ReturnByproduct()` is passed as an argument, this returns a couple of outputs `(y_and_dy(x),z(x))` (remember that `z(x)` is not differentiated so `dz(x)` doesn't exist).

We compute the Jacobian-vector product `Jv` by solving `Au = Bv` and setting `Jv = u` (see [`ImplicitFunction`](@ref) for the definition of `A` and `B`).
Keyword arguments are given to both `implicit.forward` and `implicit.conditions`.
"""
function (implicit::ImplicitFunction)(
    x_and_dx::AbstractArray{Dual{T,R,N}}, ::ReturnByproduct; kwargs...
) where {T,R,N}
    @unpack conditions, linear_solver = implicit

    x = value.(x_and_dx)
    y, z = implicit(x, ReturnByproduct(); kwargs...)
    n, m = length(x), length(y)

    backend = ForwardDiffBackend()
    pfA = pushforward_function(backend, _y -> conditions(x, _y, z; kwargs...), y)
    pfB = pushforward_function(backend, _x -> conditions(_x, y, z; kwargs...), x)

    A_op = LinearOperator(R, m, m, false, false, PushforwardMul!(pfA, size(y)))
    B_op = LinearOperator(R, m, n, false, false, PushforwardMul!(pfB, size(x)))
    A_op_presolved = presolve(linear_solver, A_op, y)

    dy = ntuple(Val(N)) do k
        dₖx_vec = vec(partials.(x_and_dx, k))
        Bdx = vec(similar(y))
        mul!(Bdx, B_op, dₖx_vec)
        dₖy_vec = solve(linear_solver, A_op_presolved, Bdx)
        lmul!(-one(R), dₖy_vec)
        reshape(dₖy_vec, size(y))
    end

    y_and_dy = let y = y, dy = dy
        y_and_dy_vec = map(eachindex(y)) do i
            Dual{T}(y[i], Partials(ntuple(k -> dy[k][i], Val(N))))
        end
        reshape(y_and_dy_vec, size(y))
    end

    return y_and_dy, z
end

function (implicit::ImplicitFunction)(
    x_and_dx::AbstractArray{Dual{T,R,N}}; kwargs...
) where {T,R,N}
    y_and_dy, z = implicit(x_and_dx, ReturnByproduct(); kwargs...)
    return y_and_dy
end

end
