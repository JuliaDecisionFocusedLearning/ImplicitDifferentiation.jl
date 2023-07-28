module ImplicitDifferentiationStaticArraysExt

@static if isdefined(Base, :get_extension)
    using StaticArrays: StaticArray, MMatrix
else
    using ..StaticArrays: StaticArray, MMatrix
end

import ImplicitDifferentiation: ImplicitDifferentiation, DirectLinearSolver
using LinearAlgebra: lu, mul!

function ImplicitDifferentiation.presolve(
    ::DirectLinearSolver, A, y::StaticArray{S,T,N}
) where {S,T,N}
    m = length(y)
    A_static = zero(MMatrix{m,m,T})
    for i in axes(A_static, 2)
        v = vec(similar(y))
        v .= zero(T)
        v[i] = one(T)
        mul!(@view(A_static[:, i]), A, v)
    end
    return lu(A_static)
end

end
