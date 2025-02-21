struct Switch12{F}
    f::F
end

function (s12::Switch12)(arg1, arg2, other_args::Vararg{Any,N}) where {N}
    return s12.f(arg2, arg1, other_args...)
end

function prepare_A(
    conditions, backend, x::AbstractVector, y::AbstractVector, z, args...; lazy
)
    contexts = (Constant(x), Constant(z), map(Constant, args)...)
    if lazy
        return prepare_pushforward(
            Switch12(conditions), backend, y, (zero(y),), contexts...
        )
    else
        return prepare_jacobian(Switch12(conditions), backend, y, contexts...)
    end
end

function prepare_Aᵀ(
    conditions, backend, x::AbstractVector, y::AbstractVector, z, args...; lazy
)
    contexts = (Constant(x), Constant(z), map(Constant, args)...)
    if lazy
        return prepare_pullback(Switch12(conditions), backend, y, (zero(y),), contexts...)
    else
        return prepare_jacobian(Switch12(conditions), backend, y, contexts...)
    end
end

function prepare_B(
    conditions, backend, x::AbstractVector, y::AbstractVector, z, args...; lazy
)
    contexts = (Constant(y), Constant(z), map(Constant, args)...)
    if lazy
        return prepare_pushforward(conditions, backend, x, (zero(x),), contexts...)
    else
        return prepare_jacobian(conditions, backend, x, contexts...)
    end
end

function prepare_Bᵀ(
    conditions, backend, x::AbstractVector, y::AbstractVector, z, args...; lazy
)
    contexts = (Constant(y), Constant(z), map(Constant, args)...)
    if lazy
        return prepare_pullback(conditions, backend, x, (zero(y),), contexts...)
    else
        return prepare_jacobian(conditions, backend, x, contexts...)
    end
end
