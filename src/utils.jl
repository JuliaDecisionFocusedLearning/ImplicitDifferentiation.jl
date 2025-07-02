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
