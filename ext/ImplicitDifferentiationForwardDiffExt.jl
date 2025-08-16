module ImplicitDifferentiationForwardDiffExt

using ADTypes: AutoForwardDiff
using ForwardDiff: Dual, Partials, partials, value
using ImplicitDifferentiation:
    ImplicitFunction,
    ImplicitFunctionPreparation,
    IterativeLeastSquaresSolver,
    build_A,
    build_Aᵀ,
    build_B

function (implicit::ImplicitFunction)(
    prep::ImplicitFunctionPreparation{R}, x_and_dx::AbstractArray{Dual{T,R,N}}, args...
) where {T,R,N}
    (; conditions, linear_solver) = implicit
    x = value.(x_and_dx)
    y, z = implicit(x, args...)
    c = conditions(x, y, z, args...)
    y0 = zero(y)

    suggested_backend = AutoForwardDiff()
    A = build_A(implicit, prep, x, y, z, c, args...; suggested_backend)
    B = build_B(implicit, prep, x, y, z, c, args...; suggested_backend)
    Aᵀ = if linear_solver isa IterativeLeastSquaresSolver
        build_Aᵀ(implicit, prep, x, y, z, c, args...; suggested_backend)
    else
        nothing
    end

    dX = ntuple(k -> partials.(x_and_dx, k), Val(N))
    dC = map(B, dX)
    dY = map(dC) do dₖc
        linear_solver(A, Aᵀ, -dₖc, y0)
    end
    y_and_dy = map(y, LinearIndices(y)) do yi, i
        Dual{T}(yi, Partials(ntuple(k -> dY[k][i], Val(N))))
    end

    return y_and_dy, z
end

function (implicit::ImplicitFunction)(
    x_and_dx::AbstractArray{Dual{T,R,N}}, args...
) where {T,R,N}
    prep = ImplicitFunctionPreparation(R)
    return implicit(prep, x_and_dx, args...)
end

end
