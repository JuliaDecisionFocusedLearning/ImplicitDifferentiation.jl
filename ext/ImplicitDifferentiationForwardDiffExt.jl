module ImplicitDifferentiationForwardDiffExt

using ADTypes: AutoForwardDiff
using ForwardDiff: Dual, Partials, partials, value
using ImplicitDifferentiation: ImplicitFunction, build_A, build_B

function (implicit::ImplicitFunction)(
    x_and_dx::AbstractArray{Dual{T,R,N}}, args...
) where {T,R,N}
    x = value.(x_and_dx)
    y, z = implicit(x, args...)

    suggested_backend = AutoForwardDiff()
    A = build_A(implicit, x, y, z, args...; suggested_backend)
    B = build_B(implicit, x, y, z, args...; suggested_backend)

    dX = ntuple(Val(N)) do k
        partials.(x_and_dx, k)
    end
    dC_mat = mapreduce(hcat, dX) do dₖx
        dₖx_vec = vec(dₖx)
        dₖc_vec = B(dₖx_vec)
        return dₖc_vec
    end
    dY_mat = implicit.linear_solver(A, -dC_mat)

    y_and_dy = map(LinearIndices(y)) do i
        Dual{T}(y[i], Partials(ntuple(k -> dY_mat[i, k], Val(N))))
    end

    return y_and_dy, z
end

end
