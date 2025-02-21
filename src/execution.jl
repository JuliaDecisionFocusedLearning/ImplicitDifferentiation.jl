struct PushforwardOperator!{F,P,B,X,C}
    f::F
    prep::P
    backend::B
    x::X
    contexts::C
end

struct PullbackOperator!{F,P,B,X,C}
    f::F
    prep::P
    backend::B
    x::X
    contexts::C
end

function (po::PushforwardOperator!)(res, v)
    (; f, backend, x, contexts, prep) = po
    pushforward!(f, (res,), prep, backend, x, (v,), contexts...)
    return res
end

function (po::PullbackOperator!)(res, v)
    (; f, backend, x, contexts, prep) = po
    pullback!(f, (res,), prep, backend, x, (v,), contexts...)
    return res
end

function build_A(
    implicit::ImplicitFunction{lazy}, x::AbstractVector, y::AbstractVector, z, args...;
) where {lazy}
    (; conditions, backend, prep_A) = implicit
    contexts = (Constant(x), Constant(z), map(Constant, args)...)
    if lazy
        prep_A_same = prepare_pushforward_same_point(
            Switch12(conditions), prep_A, backend, y, (zero(y),), contexts...
        )
        return LinearOperator(
            eltype(y),
            length(y),
            length(y),
            false,
            false,
            PushforwardOperator!(Switch12(conditions), prep_A_same, backend, y, contexts),
            typeof(y),
        )
    else
        J = jacobian(Switch12(conditions), prep_A, backend, y, contexts...)
        return factorize(J)
    end
end

function build_Aᵀ(
    implicit::ImplicitFunction{lazy}, x::AbstractVector, y::AbstractVector, z, args...;
) where {lazy}
    (; conditions, backend, prep_Aᵀ) = implicit
    contexts = (Constant(x), Constant(z), map(Constant, args)...)
    if lazy
        prep_Aᵀ_same = prepare_pullback_same_point(
            Switch12(conditions), prep_Aᵀ, backend, y, (zero(y),), contexts...
        )
        return LinearOperator(
            eltype(y),
            length(y),
            length(y),
            false,
            false,
            PullbackOperator!(Switch12(conditions), prep_Aᵀ_same, backend, y, contexts),
            typeof(y),
        )
    else
        Jᵀ = transpose(jacobian(Switch12(conditions), prep_Aᵀ, backend, y, contexts...))
        return factorize(Jᵀ)
    end
end

function build_B(
    implicit::ImplicitFunction{lazy}, x::AbstractVector, y::AbstractVector, z, args...;
) where {lazy}
    (; conditions, backend, prep_B) = implicit
    contexts = (Constant(y), Constant(z), map(Constant, args)...)
    if lazy
        prep_B_same = prepare_pushforward_same_point(
            conditions, prep_B, backend, x, (zero(x),), contexts...
        )
        return LinearOperator(
            eltype(y),
            length(y),
            length(x),
            false,
            false,
            PushforwardOperator!(conditions, prep_B_same, backend, x, contexts),
            typeof(x),
        )
    else
        return jacobian(conditions, prep_B, backend, x, contexts...)
    end
end

function build_Bᵀ(
    implicit::ImplicitFunction{lazy}, x::AbstractVector, y::AbstractVector, z, args...
) where {lazy}
    (; conditions, backend, prep_Bᵀ) = implicit
    contexts = (Constant(y), Constant(z), map(Constant, args)...)
    if lazy
        prep_Bᵀ_same = prepare_pullback_same_point(
            conditions, prep_Bᵀ, backend, x, (zero(y),), contexts...
        )
        return LinearOperator(
            eltype(y),
            length(x),
            length(y),
            false,
            false,
            PullbackOperator!(conditions, prep_Bᵀ_same, backend, x, contexts),
            typeof(x),
        )
    else
        return transpose(jacobian(conditions, prep_Bᵀ, backend, x, contexts...))
    end
end
