struct Switch12{F}
    f::F
end

function (s12::Switch12)(arg1, arg2, other_args::Vararg{Any,N}) where {N}
    return s12.f(arg2, arg1, other_args...)
end

function prepare_A(
    ::MatrixRepresentation,
    x::AbstractVector,
    y::AbstractVector,
    z,
    args...;
    conditions,
    backend::AbstractADType,
)
    contexts = (Constant(x), Constant(z), map(Constant, args)...)
    return prepare_jacobian(Switch12(conditions), backend, y, contexts...)
end

function prepare_A(
    ::OperatorRepresentation,
    x::AbstractVector,
    y::AbstractVector,
    z,
    args...;
    conditions,
    backend::AbstractADType,
)
    contexts = (Constant(x), Constant(z), map(Constant, args)...)
    return prepare_pushforward(Switch12(conditions), backend, y, (zero(y),), contexts...)
end

function prepare_Aᵀ(
    ::MatrixRepresentation,
    x::AbstractVector,
    y::AbstractVector,
    z,
    args...;
    conditions,
    backend::AbstractADType,
)
    contexts = (Constant(x), Constant(z), map(Constant, args)...)
    return prepare_jacobian(Switch12(conditions), backend, y, contexts...)
end

function prepare_Aᵀ(
    ::OperatorRepresentation,
    x::AbstractVector,
    y::AbstractVector,
    z,
    args...;
    conditions,
    backend::AbstractADType,
)
    contexts = (Constant(x), Constant(z), map(Constant, args)...)
    return prepare_pullback(Switch12(conditions), backend, y, (zero(y),), contexts...)
end

function prepare_B(
    ::MatrixRepresentation,
    x::AbstractVector,
    y::AbstractVector,
    z,
    args...;
    conditions,
    backend::AbstractADType,
)
    contexts = (Constant(y), Constant(z), map(Constant, args)...)
    return prepare_jacobian(conditions, backend, x, contexts...)
end

function prepare_B(
    ::OperatorRepresentation,
    x::AbstractVector,
    y::AbstractVector,
    z,
    args...;
    conditions,
    backend::AbstractADType,
)
    contexts = (Constant(y), Constant(z), map(Constant, args)...)
    return prepare_pushforward(conditions, backend, x, (zero(x),), contexts...)
end

function prepare_Bᵀ(
    ::MatrixRepresentation,
    x::AbstractVector,
    y::AbstractVector,
    z,
    args...;
    conditions,
    backend::AbstractADType,
)
    contexts = (Constant(y), Constant(z), map(Constant, args)...)
    return prepare_jacobian(conditions, backend, x, contexts...)
end

function prepare_Bᵀ(
    ::OperatorRepresentation,
    x::AbstractVector,
    y::AbstractVector,
    z,
    args...;
    conditions,
    backend::AbstractADType,
)
    contexts = (Constant(y), Constant(z), map(Constant, args)...)
    return prepare_pullback(conditions, backend, x, (zero(y),), contexts...)
end
