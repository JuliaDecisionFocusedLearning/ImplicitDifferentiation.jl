module ImplicitDifferentiationStaticArraysExt

@static if isdefined(Base, :get_extension)
    using StaticArrays: StaticArray, MMatrix, StaticVector
else
    using ..StaticArrays: StaticArray, MMatrix, StaticVector
end

import ImplicitDifferentiation: ImplicitDifferentiation, DirectLinearSolver
using Krylov: Krylov
using LinearAlgebra: lu, mul!

function ImplicitDifferentiation.presolve(::DirectLinearSolver, A, y::StaticArray)
    T = eltype(A)
    m = length(y)
    A_static = zero(MMatrix{m,m,T})
    v = vec(similar(y, T))
    for i in axes(A_static, 2)
        v .= zero(T)
        v[i] = one(T)
        mul!(@view(A_static[:, i]), A, v)
    end
    return lu(A_static)
end

"""
    Krylov.ktypeof(::StaticVector)

!!! danger "Danger"
    This is type piracy.
"""
Krylov.ktypeof(::StaticVector{S,T}) where {S,T} = Vector{T}  # TODO: type piracy

end
