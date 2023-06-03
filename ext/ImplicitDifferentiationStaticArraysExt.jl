module ImplicitDifferentiationStaticArraysExt

@static if isdefined(Base, :get_extension)
    using StaticArrays: StaticVector, MMatrix, LU
else
    using ..StaticArrays: StaticVector, MMatrix, LU
end

import ImplicitDifferentiation: direct_presolver, default_presolver, auto_linear_solver
using ImplicitDifferentiation: direct_linear_solver
using LinearAlgebra: lu, mul!, I

auto_linear_solver(A::LU, b) = direct_linear_solver(A, b)

function direct_presolver(A, ::StaticVector{N1}, y::StaticVector{N2}) where {N1,N2}
    _A = zero(MMatrix{N1,N2,eltype(A)})
    for i in 1:size(_A, 2)
        v = similar(y)
        v .= 0
        v[i] = 1
        mul!(@view(_A[:, i]), A, v)
    end
    return lu(_A)
end

function default_presolver(A, x::StaticVector{N1}, y::StaticVector{N2}) where {N1,N2}
    return direct_presolver(A, x, y)
end

end
