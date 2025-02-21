module ImplicitDifferentiationForwardDiffExt

using ADTypes: AutoForwardDiff
using ForwardDiff: Chunk, Dual, Partials, jacobian, partials, value
using ImplicitDifferentiation: ImplicitFunction, build_A, build_B

function (implicit::ImplicitFunction)(
    x_and_dx::AbstractVector{Dual{T,R,N}}, args...; kwargs...
) where {T,R,N}
    x = value.(x_and_dx)
    y, z = implicit(x, args...; kwargs...)

    A = build_A(implicit, x, y, z, args...)
    B = build_B(implicit, x, y, z, args...)

    dX = mapreduce(hcat, 1:N) do k
        partials.(x_and_dx, k)
    end
    dC = mapreduce(hcat, eachcol(dX)) do dₖx
        B * dₖx
    end
    dY = implicit.linear_solver(A, -dC)

    y_and_dy = map(eachindex(y)) do i
        Dual{T}(y[i], Partials(ntuple(k -> dY[i, k], Val(N))))
    end

    return y_and_dy, z
end

end
