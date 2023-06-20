module ImplicitDifferentiationStaticArraysExt

@static if isdefined(Base, :get_extension)
    using StaticArrays: StaticArray, MMatrix
else
    using ..StaticArrays: StaticArray, MMatrix
end

import ImplicitDifferentiation: ImplicitDifferentiation, DirectLinearSolver
using LinearAlgebra: lu, mul!

_prodsize(::Type{Tuple{}}) = 1
_prodsize(::Type{Tuple{N1}}) where {N1} = N1
_prodsize(::Type{Tuple{N1,N2}}) where {N1,N2} = N1 * N2
_prodsize(::Type{Tuple{N1,N2,N3}}) where {N1,N2,N3} = N1 * N2 * N3
_prodsize(::Type{Tuple{N1,N2,N3,N4}}) where {N1,N2,N3,N4} = N1 * N2 * N3 * N4

function ImplicitDifferentiation.presolve(
    ::DirectLinearSolver, A, y::StaticArray{S,T,N}
) where {S,T,N}
    S_prod = _prodsize(S)
    A_static = zero(MMatrix{S_prod,S_prod,T})
    for i in axes(A_static, 2)
        v = vec(similar(y))
        v .= zero(T)
        v[i] = one(T)
        mul!(@view(A_static[:, i]), A, v)
    end
    return lu(A_static)
end

end
