module ImplicitDifferentiationForwardDiffExt

using ADTypes: AutoForwardDiff
using ForwardDiff: Chunk, Dual, Partials, jacobian, partials, value
using ImplicitDifferentiation: ImplicitFunction, build_A, build_B, get_byproduct, get_output

chunksize(::Chunk{N}) where {N} = N

function (implicit::ImplicitFunction)(
    x_and_dx::AbstractVector{Dual{T,R,N}}, args...; kwargs...
) where {T,R,N}
    x = value.(x_and_dx)
    y_or_yz = implicit(x, args...; kwargs...)
    y = get_output(y_or_yz)

    suggested_backend = AutoForwardDiff{1,Nothing}(nothing)
    A = build_A(implicit, x, y_or_yz, args...; suggested_backend, kwargs...)
    B = build_B(implicit, x, y_or_yz, args...; suggested_backend, kwargs...)

    dy = ntuple(Val(N)) do k
        dₖx = partials.(x_and_dx, k)
        dₖc = B * dₖx
        dₖy = implicit.linear_solver(A, -dₖc)
        return dₖy
    end

    y_and_dy = map(eachindex(y)) do i
        Dual{T}(y[i], Partials(ntuple(k -> dy[k][i], Val(N))))
    end

    if y_or_yz isa Tuple
        return y_and_dy, get_byproduct(y_or_yz)
    else
        return y_and_dy
    end
end

end
