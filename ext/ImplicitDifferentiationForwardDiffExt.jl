module ImplicitDifferentiationForwardDiffExt

using ADTypes: AutoForwardDiff
using ForwardDiff: Chunk, Dual, Partials, jacobian, partials, value
using ImplicitDifferentiation: ImplicitFunction, build_A, build_B, byproduct, output

chunksize(::Chunk{N}) where {N} = N

function (implicit::ImplicitFunction)(
    x_and_dx::AbstractVector{Dual{T,R,N}}, args...; kwargs...
) where {T,R,N}
    x = value.(x_and_dx)
    y_or_yz = implicit(x, args...; kwargs...)
    y = output(y_or_yz)

    A = build_A(
        implicit,
        x,
        y_or_yz,
        args...;
        suggested_backend=AutoForwardDiff(; tag=T(), chunksize=chunksize(Chunk(y))),
        kwargs...,
    )
    B = build_B(
        implicit,
        x,
        y_or_yz,
        args...;
        suggested_backend=AutoForwardDiff(; tag=T(), chunksize=chunksize(Chunk(x))),
        kwargs...,
    )

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

    if y_or_yz isa Tuple
        return y_and_dy, byproduct(y_or_yz)
    else
        return y_and_dy
    end
end

end
