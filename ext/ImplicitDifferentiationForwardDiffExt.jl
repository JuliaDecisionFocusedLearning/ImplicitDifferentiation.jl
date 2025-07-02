module ImplicitDifferentiationForwardDiffExt

using ADTypes: AutoForwardDiff
using ForwardDiff: Dual, Partials, partials, value
using ImplicitDifferentiation:
    ImplicitFunction, ImplicitFunctionPreparation, build_A, build_B

function (implicit::ImplicitFunction)(
    prep::ImplicitFunctionPreparation, x_and_dx::AbstractArray{Dual{T,R,N}}, args...
) where {T,R,N}
    x = value.(x_and_dx)
    y, z = implicit(x, args...)
    c = implicit.conditions(x, y, z, args...)

    suggested_backend = AutoForwardDiff()
    A = build_A(implicit, prep, x, y, z, c, args...; suggested_backend)
    B = build_B(implicit, prep, x, y, z, c, args...; suggested_backend)

    dX = ntuple(Val(N)) do k
        partials.(x_and_dx, k)
    end
    dC_vec = map(dX) do dₖx
        dₖx_vec = vec(dₖx)
        dₖc_vec = B(dₖx_vec)
        return dₖc_vec
    end
    dY = map(dC_vec) do dₖc_vec
        dₖy_vec = implicit.linear_solver(A, -dₖc_vec)
        dₖy = reshape(dₖy_vec, size(y))
        return dₖy
    end

    y_and_dy = map(y, LinearIndices(y)) do yi, i
        Dual{T}(yi, Partials(ntuple(k -> dY[k][i], Val(N))))
    end

    return y_and_dy, z
end

end
