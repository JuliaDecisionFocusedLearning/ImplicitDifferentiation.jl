struct JVP{F,P,B,I,C}
    f::F
    prep::P
    backend::B
    input::I
    contexts::C
end

struct VJP{F,P,B,I,C}
    f::F
    prep::P
    backend::B
    input::I
    contexts::C
end

function (po::JVP)(v)
    (; f, backend, input, contexts, prep) = po
    res = pushforward(f, prep, backend, input, (v,), contexts...)
    return only(res)
end

function (po::VJP)(v)
    (; f, backend, input, contexts, prep) = po
    res = pullback(f, prep, backend, input, (v,), contexts...)
    return only(res)
end

## A

function build_A(
    implicit::ImplicitFunction,
    prep::ImplicitFunctionPreparation,
    x::AbstractArray,
    y::AbstractArray,
    z,
    c,
    args...;
    suggested_backend::AbstractADType,
)
    return build_A_aux(
        implicit.representation, implicit, prep, x, y, z, c, args...; suggested_backend
    )
end

function build_A_aux(
    ::MatrixRepresentation, implicit, prep, x, y, z, c, args...; suggested_backend
)
    (; conditions, linear_solver, backends) = implicit
    (; prep_A) = prep
    actual_backend = isnothing(backends) ? suggested_backend : backends.y
    contexts = (Constant(x), Constant(z), map(Constant, args)...)
    f = Switch12(conditions)
    if isnothing(prep_A)
        A = jacobian(f, actual_backend, y, contexts...)
    else
        A = jacobian(f, prep_A, actual_backend, y, contexts...)
    end
    if linear_solver isa DirectLinearSolver
        return factorize(A)
    else
        return A
    end
end

function build_A_aux(
    ::OperatorRepresentation, implicit, prep, x, y, z, c, args...; suggested_backend
)
    (; conditions, backends) = implicit
    (; prep_A) = prep
    actual_backend = isnothing(backends) ? suggested_backend : backends.y
    contexts = (Constant(x), Constant(z), map(Constant, args)...)
    f = Switch12(conditions)
    if isnothing(prep_A)
        prep_A_same = prepare_pushforward_same_point(
            f, actual_backend, y, (zero(y),), contexts...; strict=implicit.strict
        )
    else
        prep_A_same = prepare_pushforward_same_point(
            f, prep_A, actual_backend, y, (zero(y),), contexts...
        )
    end
    A = JVP(f, prep_A_same, actual_backend, y, contexts)
    return A
end

## Aᵀ

function build_Aᵀ(
    implicit::ImplicitFunction,
    prep::ImplicitFunctionPreparation,
    x::AbstractArray,
    y::AbstractArray,
    z,
    c,
    args...;
    suggested_backend::AbstractADType,
)
    return build_Aᵀ_aux(
        implicit.representation, implicit, prep, x, y, z, c, args...; suggested_backend
    )
end

function build_Aᵀ_aux(
    ::MatrixRepresentation, implicit, prep, x, y, z, c, args...; suggested_backend
)
    (; conditions, linear_solver, backends) = implicit
    (; prep_Aᵀ) = prep
    actual_backend = isnothing(backends) ? suggested_backend : backends.y
    contexts = (Constant(x), Constant(z), map(Constant, args)...)
    f = Switch12(conditions)
    if isnothing(prep_Aᵀ)
        Aᵀ = transpose(jacobian(f, actual_backend, y, contexts...))
    else
        Aᵀ = transpose(jacobian(f, prep_Aᵀ, actual_backend, y, contexts...))
    end
    if linear_solver isa DirectLinearSolver
        return factorize(Aᵀ)
    else
        return Aᵀ
    end
end

function build_Aᵀ_aux(
    ::OperatorRepresentation, implicit, prep, x, y, z, c, args...; suggested_backend
)
    (; conditions, backends) = implicit
    (; prep_Aᵀ) = prep
    actual_backend = isnothing(backends) ? suggested_backend : backends.y
    contexts = (Constant(x), Constant(z), map(Constant, args)...)
    f = Switch12(conditions)
    if isnothing(prep_Aᵀ)
        prep_Aᵀ_same = prepare_pullback_same_point(
            f, actual_backend, y, (zero(c),), contexts...; strict=implicit.strict
        )
    else
        prep_Aᵀ_same = prepare_pullback_same_point(
            f, prep_Aᵀ, actual_backend, y, (zero(c),), contexts...
        )
    end
    A = VJP(f, prep_Aᵀ_same, actual_backend, y, contexts)
    return A
end

## B

function build_B(
    implicit::ImplicitFunction,
    prep::ImplicitFunctionPreparation,
    x::AbstractArray,
    y::AbstractArray,
    z,
    c,
    args...;
    suggested_backend::AbstractADType,
)
    (; conditions, backends) = implicit
    (; prep_B) = prep
    actual_backend = isnothing(backends) ? suggested_backend : backends.x
    contexts = (Constant(y), Constant(z), map(Constant, args)...)
    if isnothing(prep_B)
        prep_B_same = prepare_pushforward_same_point(
            conditions, actual_backend, x, (zero(x),), contexts...; strict=implicit.strict
        )
    else
        prep_B_same = prepare_pushforward_same_point(
            conditions, prep_B, actual_backend, x, (zero(x),), contexts...
        )
    end
    B = JVP(conditions, prep_B_same, actual_backend, x, contexts)
    return B
end

## Bᵀ

function build_Bᵀ(
    implicit::ImplicitFunction,
    prep::ImplicitFunctionPreparation,
    x::AbstractArray,
    y::AbstractArray,
    z,
    c,
    args...;
    suggested_backend::AbstractADType,
)
    (; conditions, backends) = implicit
    (; prep_Bᵀ) = prep
    actual_backend = isnothing(backends) ? suggested_backend : backends.x
    contexts = (Constant(y), Constant(z), map(Constant, args)...)
    if isnothing(prep_Bᵀ)
        prep_Bᵀ_same = prepare_pullback_same_point(
            conditions, actual_backend, x, (zero(c),), contexts...; strict=implicit.strict
        )
    else
        prep_Bᵀ_same = prepare_pullback_same_point(
            conditions, prep_Bᵀ, actual_backend, x, (zero(c),), contexts...
        )
    end
    Bᵀ = VJP(conditions, prep_Bᵀ_same, actual_backend, x, contexts)
    return Bᵀ
end
