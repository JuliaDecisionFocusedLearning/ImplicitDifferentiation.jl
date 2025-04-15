"""
    Switch12

Represent a function which behaves like `f`, except that the first and second arguments are switched:
    f(a1, a2, a3) = b
becomes
    g(a2, a1, a3) = f(a1, a2, a3)
"""
struct Switch12{F}
    f::F
end

function (s12::Switch12)(arg1, arg2, other_args::Vararg{Any,N}) where {N}
    return s12.f(arg2, arg1, other_args...)
end

"""
    VecToVec

Represent a function which behaves like `f`, except that the first argument is expected as a vector, and the return is converted to a vector:
    f(a1, a2, a3) = b
becomes
    g(a1_vec, a2, a3) = vec(f(reshape(a1_vec, size(a1)), a2, a3))
"""
struct VecToVec{F,N}
    f::F
    arg1_size::NTuple{N,Int}
end

VecToVec(f::F, arg1_example::AbstractArray) where {F} = VecToVec(f, size(arg1_example))

function (v2v::VecToVec)(arg1_vec::AbstractVector, other_args::Vararg{Any,N}) where {N}
    arg1 = reshape(arg1_vec, v2v.arg1_size)
    res = v2v.f(arg1, other_args...)
    return vec(res)
end
