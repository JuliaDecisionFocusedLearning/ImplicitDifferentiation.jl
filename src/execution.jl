struct JVP!{F,P,B,I,V,C}
    f::F
    prep::P
    backend::B
    input::I
    v_buffer::V
    contexts::C
end

struct VJP!{F,P,B,I,V,C}
    f::F
    prep::P
    backend::B
    input::I
    v_buffer::V
    contexts::C
end

function (po::JVP!)(res::AbstractVector, v_wrongtype::AbstractVector)
    (; f, backend, input, v_buffer, contexts, prep) = po
    if typeof(v_buffer) == typeof(v_wrongtype)
        v = v_wrongtype
    else
        copyto!(v_buffer, v_wrongtype)
        v = v_buffer
    end
    pushforward!(f, (res,), prep, backend, input, (v,), contexts...)
    return res
end

function (po::VJP!)(res::AbstractVector, v_wrongtype::AbstractVector)
    (; f, backend, input, v_buffer, contexts, prep) = po
    if typeof(v_buffer) == typeof(v_wrongtype)
        v = v_wrongtype
    else
        copyto!(v_buffer, v_wrongtype)
        v = v_buffer
    end
    pullback!(f, (res,), prep, backend, input, (v,), contexts...)
    return res
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
    (; conditions, backends) = implicit
    (; prep_A) = prep
    actual_backend = isnothing(backends) ? suggested_backend : backends.y
    contexts = (Constant(x), Constant(z), map(Constant, args)...)
    if isnothing(prep_A)
        A = jacobian(Switch12(conditions), actual_backend, y, contexts...)
    else
        A = jacobian(Switch12(conditions), prep_A, actual_backend, y, contexts...)
    end
    return factorize(A)
end

function build_A_aux(
    ::OperatorRepresentation{package,symmetric,hermitian,posdef},
    implicit,
    prep,
    x,
    y,
    z,
    c,
    args...;
    suggested_backend,
) where {package,symmetric,hermitian,posdef}
    T = Base.promote_eltype(x, y, c)
    (; conditions, backends) = implicit
    (; prep_A) = prep
    actual_backend = isnothing(backends) ? suggested_backend : backends.y
    contexts = (Constant(x), Constant(z), map(Constant, args)...)
    f_vec = VecToVec(Switch12(conditions), y)
    y_vec = vec(y)
    dy_vec = vec(zero(y))
    if isnothing(prep_A)
        prep_A_same = prepare_pushforward_same_point(
            f_vec, actual_backend, y_vec, (dy_vec,), contexts...; strict=implicit.strict
        )
    else
        prep_A_same = prepare_pushforward_same_point(
            f_vec, prep_A, actual_backend, y_vec, (dy_vec,), contexts...
        )
    end
    prod! = JVP!(f_vec, prep_A_same, actual_backend, y_vec, dy_vec, contexts)
    if package == :LinearOperators
        return LinearOperator(T, length(c), length(y), symmetric, hermitian, prod!)
    elseif package == :LinearMaps
        return FunctionMap{T}(
            prod!,
            length(c),
            length(y);
            ismutating=true,
            issymmetric=symmetric,
            ishermitian=hermitian,
            isposdef=posdef,
        )
    end
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
    (; conditions, backends) = implicit
    (; prep_Aᵀ) = prep
    actual_backend = isnothing(backends) ? suggested_backend : backends.y
    contexts = (Constant(x), Constant(z), map(Constant, args)...)
    if isnothing(prep_Aᵀ)
        Aᵀ = transpose(jacobian(Switch12(conditions), actual_backend, y, contexts...))
    else
        Aᵀ = transpose(
            jacobian(Switch12(conditions), prep_Aᵀ, actual_backend, y, contexts...)
        )
    end
    return factorize(Aᵀ)
end

function build_Aᵀ_aux(
    ::OperatorRepresentation{package,symmetric,hermitian,posdef},
    implicit,
    prep,
    x,
    y,
    z,
    c,
    args...;
    suggested_backend,
) where {package,symmetric,hermitian,posdef}
    T = Base.promote_eltype(x, y, c)
    (; conditions, backends) = implicit
    (; prep_Aᵀ) = prep
    actual_backend = isnothing(backends) ? suggested_backend : backends.y
    contexts = (Constant(x), Constant(z), map(Constant, args)...)
    f_vec = VecToVec(Switch12(conditions), y)
    y_vec = vec(y)
    dc_vec = vec(zero(c))
    if isnothing(prep_Aᵀ)
        prep_Aᵀ_same = prepare_pullback_same_point(
            f_vec, actual_backend, y_vec, (dc_vec,), contexts...; strict=implicit.strict
        )
    else
        prep_Aᵀ_same = prepare_pullback_same_point(
            f_vec, prep_Aᵀ, actual_backend, y_vec, (dc_vec,), contexts...
        )
    end
    prod! = VJP!(f_vec, prep_Aᵀ_same, actual_backend, y_vec, dc_vec, contexts)
    if package == :LinearOperators
        return LinearOperator(T, length(y), length(c), symmetric, hermitian, prod!)
    elseif package == :LinearMaps
        return FunctionMap{T}(
            prod!,
            length(y),
            length(c);
            ismutating=true,
            issymmetric=symmetric,
            ishermitian=hermitian,
            isposdef=posdef,
        )
    end
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
    f_vec = VecToVec(conditions, x)
    x_vec = vec(x)
    dx_vec = vec(zero(x))
    if isnothing(prep_B)
        prep_B_same = prepare_pushforward_same_point(
            f_vec, actual_backend, x_vec, (dx_vec,), contexts...
        )
    else
        prep_B_same = prepare_pushforward_same_point(
            f_vec, prep_B, actual_backend, x_vec, (dx_vec,), contexts...
        )
    end
    function B_fun(dx_vec_wrongtype)
        @assert typeof(dx_vec) == typeof(dx_vec_wrongtype)
        dx_vec = dx_vec_wrongtype
        return pushforward(
            f_vec, prep_B_same, actual_backend, x_vec, (dx_vec,), contexts...
        )[1]
    end
    return B_fun
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
    f_vec = VecToVec(conditions, x)
    x_vec = vec(x)
    dc_vec = vec(zero(c))
    if isnothing(prep_Bᵀ)
        prep_Bᵀ_same = prepare_pullback_same_point(
            f_vec, actual_backend, x_vec, (dc_vec,), contexts...; strict=implicit.strict
        )
    else
        prep_Bᵀ_same = prepare_pullback_same_point(
            f_vec, prep_Bᵀ, actual_backend, x_vec, (dc_vec,), contexts...
        )
    end
    function Bᵀ_fun(dc_vec_wrongtype)
        if typeof(dc_vec) == typeof(dc_vec_wrongtype)
            dc_vec = dc_vec_wrongtype
        else
            copyto!(dc_vec, dc_vec_wrongtype)
        end
        return pullback(f_vec, prep_Bᵀ_same, actual_backend, x_vec, (dc_vec,), contexts...)[1]
    end
    return Bᵀ_fun
end
