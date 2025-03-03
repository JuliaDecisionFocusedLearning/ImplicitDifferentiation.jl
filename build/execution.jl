const SYMMETRIC = false
const HERMITIAN = false

struct JVP!{F,P,B,X,C}
    f::F
    prep::P
    backend::B
    x::X
    contexts::C
end

struct VJP!{F,P,B,X,C}
    f::F
    prep::P
    backend::B
    x::X
    contexts::C
end

function (po::JVP!)(res::AbstractVector, v::AbstractVector)
    (; f, backend, x, contexts, prep) = po
    pushforward!(f, (res,), prep, backend, x, (v,), contexts...)
    return res
end

function (po::VJP!)(res::AbstractVector, v::AbstractVector)
    (; f, backend, x, contexts, prep) = po
    pullback!(f, (res,), prep, backend, x, (v,), contexts...)
    return res
end

## A

function build_A(
    implicit::ImplicitFunction,
    x::AbstractVector,
    y::AbstractVector,
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
    prep_A_same = prepare_pushforward_same_point(
        Switch12(conditions), prep_A..., actual_backend, y, (zero(y),), contexts...
    )
    prod! = JVP!(Switch12(conditions), prep_A_same, actual_backend, y, contexts)
    return LinearOperator(
        eltype(y), length(y), length(y), SYMMETRIC, HERMITIAN, prod!, typeof(y)
    )
end

## Aᵀ

function build_Aᵀ(
    implicit::ImplicitFunction,
    x::AbstractVector,
    y::AbstractVector,
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
    prep_Aᵀ_same = prepare_pullback_same_point(
        Switch12(conditions), prep_Aᵀ..., actual_backend, y, (zero(y),), contexts...
    )
    prod! = VJP!(Switch12(conditions), prep_Aᵀ_same, actual_backend, y, contexts)
    return LinearOperator(
        eltype(y), length(y), length(y), SYMMETRIC, HERMITIAN, prod!, typeof(y)
    )
end

## B

function build_B(
    implicit::ImplicitFunction,
    x::AbstractVector,
    y::AbstractVector,
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
    prep_B_same = prepare_pushforward_same_point(
        conditions, prep_B..., actual_backend, x, (zero(x),), contexts...
    )
    prod! = JVP!(conditions, prep_B_same, actual_backend, x, contexts)
    return LinearOperator(
        eltype(y), length(y), length(x), SYMMETRIC, HERMITIAN, prod!, typeof(x)
    )
end

## Bᵀ

function build_Bᵀ(
    implicit::ImplicitFunction,
    x::AbstractVector,
    y::AbstractVector,
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
    prep_Bᵀ_same = prepare_pullback_same_point(
        conditions, prep_Bᵀ..., actual_backend, x, (zero(y),), contexts...
    )
    prod! = VJP!(conditions, prep_Bᵀ_same, actual_backend, x, contexts)
    return LinearOperator(
        eltype(y), length(x), length(y), SYMMETRIC, HERMITIAN, prod!, typeof(x)
    )
end
