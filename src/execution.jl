const SYMMETRIC = false
const HERMITIAN = false

struct JVP!{F,P,B,I,C}
    f::F
    prep::P
    backend::B
    input::I
    contexts::C
end

struct VJP!{F,P,B,I,C}
    f::F
    prep::P
    backend::B
    input::I
    contexts::C
end

function (po::JVP!)(res::AbstractVector, v::AbstractVector)
    (; f, backend, input, contexts, prep) = po
    pushforward!(f, (res,), prep, backend, input, (v,), contexts...)
    return res
end

function (po::VJP!)(res::AbstractVector, v::AbstractVector)
    (; f, backend, input, contexts, prep) = po
    pullback!(f, (res,), prep, backend, input, (v,), contexts...)
    return res
end

## A

function build_A(
    implicit::ImplicitFunction,
    x::AbstractArray,
    y::AbstractArray,
    z,
    args...;
    suggested_backend::AbstractADType,
)
    return build_A_aux(
        implicit.representation, implicit, x, y, z, args...; suggested_backend
    )
end

function build_A_aux(::MatrixRepresentation, implicit, x, y, z, args...; suggested_backend)
    (; conditions, backend, prep_A) = implicit
    actual_backend = isnothing(backend) ? suggested_backend : backend
    contexts = (Constant(x), Constant(z), map(Constant, args)...)
    A = jacobian(Switch12(conditions), prep_A..., actual_backend, y, contexts...)
    return factorize(A)
end

function build_A_aux(
    ::OperatorRepresentation, implicit, x, y, z, args...; suggested_backend
)
    (; conditions, backend, prep_A) = implicit
    actual_backend = isnothing(backend) ? suggested_backend : backend
    contexts = (Constant(x), Constant(z), map(Constant, args)...)
    f_vec = VecToVec(Switch12(conditions), y)
    y_vec = vec(y)
    dy_vec = vec(zero(y))
    prep_A_same = prepare_pushforward_same_point(
        f_vec, prep_A..., actual_backend, y_vec, (dy_vec,), contexts...
    )
    prod! = JVP!(f_vec, prep_A_same, actual_backend, y_vec, contexts)
    return LinearOperator(
        eltype(y), length(y), length(y), SYMMETRIC, HERMITIAN, prod!, typeof(y_vec)
    )
end

## Aᵀ

function build_Aᵀ(
    implicit::ImplicitFunction,
    x::AbstractArray,
    y::AbstractArray,
    z,
    args...;
    suggested_backend::AbstractADType,
)
    return build_Aᵀ_aux(
        implicit.representation, implicit, x, y, z, args...; suggested_backend
    )
end

function build_Aᵀ_aux(::MatrixRepresentation, implicit, x, y, z, args...; suggested_backend)
    (; conditions, backend, prep_Aᵀ) = implicit
    actual_backend = isnothing(backend) ? suggested_backend : backend
    contexts = (Constant(x), Constant(z), map(Constant, args)...)
    Aᵀ = transpose(
        jacobian(Switch12(conditions), prep_Aᵀ..., actual_backend, y, contexts...)
    )
    return factorize(Aᵀ)
end

function build_Aᵀ_aux(
    ::OperatorRepresentation, implicit, x, y, z, args...; suggested_backend
)
    (; conditions, backend, prep_Aᵀ) = implicit
    actual_backend = isnothing(backend) ? suggested_backend : backend
    contexts = (Constant(x), Constant(z), map(Constant, args)...)
    f_vec = VecToVec(Switch12(conditions), y)
    y_vec = vec(y)
    dc_vec = vec(zero(y))
    prep_Aᵀ_same = prepare_pullback_same_point(
        f_vec, prep_Aᵀ..., actual_backend, y_vec, (dc_vec,), contexts...
    )
    prod! = VJP!(f_vec, prep_Aᵀ_same, actual_backend, y_vec, contexts)
    return LinearOperator(
        eltype(y), length(y), length(y), SYMMETRIC, HERMITIAN, prod!, typeof(y_vec)
    )
end

## B

function build_B(
    implicit::ImplicitFunction,
    x::AbstractArray,
    y::AbstractArray,
    z,
    args...;
    suggested_backend::AbstractADType,
)
    return build_B_aux(
        implicit.representation, implicit, x, y, z, args...; suggested_backend
    )
end

function build_B_aux(::MatrixRepresentation, implicit, x, y, z, args...; suggested_backend)
    (; conditions, backend, prep_B) = implicit
    actual_backend = isnothing(backend) ? suggested_backend : backend
    contexts = (Constant(y), Constant(z), map(Constant, args)...)
    return jacobian(conditions, prep_B..., actual_backend, x, contexts...)
end

function build_B_aux(
    ::OperatorRepresentation, implicit, x, y, z, args...; suggested_backend
)
    (; conditions, backend, prep_B) = implicit
    actual_backend = isnothing(backend) ? suggested_backend : backend
    contexts = (Constant(y), Constant(z), map(Constant, args)...)
    f_vec = VecToVec(conditions, x)
    x_vec = vec(x)
    dx_vec = vec(zero(x))
    prep_B_same = prepare_pushforward_same_point(
        f_vec, prep_B..., actual_backend, x_vec, (dx_vec,), contexts...
    )
    prod! = JVP!(f_vec, prep_B_same, actual_backend, x_vec, contexts)
    return LinearOperator(
        eltype(y), length(y), length(x), SYMMETRIC, HERMITIAN, prod!, typeof(x_vec)
    )
end

## Bᵀ

function build_Bᵀ(
    implicit::ImplicitFunction,
    x::AbstractArray,
    y::AbstractArray,
    z,
    args...;
    suggested_backend::AbstractADType,
)
    return build_Bᵀ_aux(
        implicit.representation, implicit, x, y, z, args...; suggested_backend
    )
end

function build_Bᵀ_aux(::MatrixRepresentation, implicit, x, y, z, args...; suggested_backend)
    (; conditions, backend, prep_Bᵀ) = implicit
    actual_backend = isnothing(backend) ? suggested_backend : backend
    contexts = (Constant(y), Constant(z), map(Constant, args)...)
    return transpose(jacobian(conditions, prep_Bᵀ..., actual_backend, x, contexts...))
end

function build_Bᵀ_aux(
    ::OperatorRepresentation, implicit, x, y, z, args...; suggested_backend
)
    (; conditions, backend, prep_Bᵀ) = implicit
    actual_backend = isnothing(backend) ? suggested_backend : backend
    contexts = (Constant(y), Constant(z), map(Constant, args)...)
    f_vec = VecToVec(conditions, x)
    x_vec = vec(x)
    dc_vec = vec(zero(y))
    prep_Bᵀ_same = prepare_pullback_same_point(
        f_vec, prep_Bᵀ..., actual_backend, x_vec, (dc_vec,), contexts...
    )
    prod! = VJP!(f_vec, prep_Bᵀ_same, actual_backend, x_vec, contexts)
    return LinearOperator(
        eltype(y), length(x), length(y), SYMMETRIC, HERMITIAN, prod!, typeof(x_vec)
    )
end
