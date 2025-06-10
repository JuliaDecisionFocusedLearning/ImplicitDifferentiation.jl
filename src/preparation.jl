function prepare_A(
    ::MatrixRepresentation,
    x::AbstractArray,
    y::AbstractArray,
    z,
    c,
    args...;
    conditions,
    backend::AbstractADType,
    strict::Val,
)
    contexts = (Constant(x), Constant(z), map(Constant, args)...)
    return prepare_jacobian(Switch12(conditions), backend, y, contexts...; strict)
end

function prepare_A(
    ::OperatorRepresentation,
    x::AbstractArray,
    y::AbstractArray,
    z,
    c,
    args...;
    conditions,
    backend::AbstractADType,
    strict::Val,
)
    contexts = (Constant(x), Constant(z), map(Constant, args)...)
    f_vec = VecToVec(Switch12(conditions), y)
    y_vec = vec(y)
    dy_vec = vec(zero(y))
    return prepare_pushforward(f_vec, backend, y_vec, (dy_vec,), contexts...; strict)
end

function prepare_Aᵀ(
    ::MatrixRepresentation,
    x::AbstractArray,
    y::AbstractArray,
    z,
    c,
    args...;
    conditions,
    backend::AbstractADType,
    strict::Val,
)
    contexts = (Constant(x), Constant(z), map(Constant, args)...)
    return prepare_jacobian(Switch12(conditions), backend, y, contexts...; strict)
end

function prepare_Aᵀ(
    ::OperatorRepresentation,
    x::AbstractArray,
    y::AbstractArray,
    z,
    c,
    args...;
    conditions,
    backend::AbstractADType,
    strict::Val,
)
    contexts = (Constant(x), Constant(z), map(Constant, args)...)
    f_vec = VecToVec(Switch12(conditions), y)
    y_vec = vec(y)
    dc_vec = vec(zero(c))
    return prepare_pullback(f_vec, backend, y_vec, (dc_vec,), contexts...; strict)
end

function prepare_B(
    ::AbstractRepresentation,
    x::AbstractArray,
    y::AbstractArray,
    z,
    c,
    args...;
    conditions,
    backend::AbstractADType,
    strict::Val,
)
    contexts = (Constant(y), Constant(z), map(Constant, args)...)
    f_vec = VecToVec(conditions, x)
    x_vec = vec(x)
    dx_vec = vec(zero(x))
    return prepare_pushforward(f_vec, backend, x_vec, (dx_vec,), contexts...; strict)
end

function prepare_Bᵀ(
    ::AbstractRepresentation,
    x::AbstractArray,
    y::AbstractArray,
    z,
    c,
    args...;
    conditions,
    backend::AbstractADType,
    strict::Val,
)
    contexts = (Constant(y), Constant(z), map(Constant, args)...)
    f_vec = VecToVec(conditions, x)
    x_vec = vec(x)
    dc_vec = vec(zero(c))
    return prepare_pullback(f_vec, backend, x_vec, (dc_vec,), contexts...; strict)
end
