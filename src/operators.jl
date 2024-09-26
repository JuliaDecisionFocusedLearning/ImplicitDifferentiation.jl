struct ConditionsX{C,K}
    conditions::C
    kwargs::K
end

struct ConditionsY{C,K}
    conditions::C
    kwargs::K
end

function (cx::ConditionsX)(x, y, args...)
    return cx.conditions(x, y, args...; cx.kwargs...)
end

function (cy::ConditionsY)(y, x, args...)  # order switch
    return cy.conditions(x, y, args...; cy.kwargs...)
end

struct PushforwardOperator!{F,P,B,X,C,R}
    f::F
    prep::P
    backend::B
    x::X
    contexts::C
    res_backup::R
end

struct PullbackOperator!{F,P,B,X,C,R}
    f::F
    prep::P
    backend::B
    x::X
    contexts::C
    res_backup::R
end

function PushforwardOperator!(f, prep, backend, x, contexts)
    res_backup = similar(f(x, map(unwrap, contexts)...))
    return PushforwardOperator!(f, prep, backend, x, contexts, res_backup)
end

function PullbackOperator!(f, prep, backend, x, contexts)
    res_backup = similar(x)
    return PullbackOperator!(f, prep, backend, x, contexts, res_backup)
end

function (po::PushforwardOperator!)(res, v, α, β)
    (; f, backend, x, contexts, prep, res_backup) = po
    if iszero(β)
        pushforward!(f, (res,), prep, backend, x, (v,), contexts...)
        if !isone(α)
            res .*= α
        end
    else
        copyto!(res_backup, res)
        pushforward!(f, (res,), prep, backend, x, (v,), contexts...)
        axpby!(β, res_backup, α, res)
    end
    return res
end

function (po::PullbackOperator!)(res, v, α, β)
    (; f, backend, x, contexts, prep, res_backup) = po
    if iszero(β)
        pullback!(f, (res,), prep, backend, x, (v,), contexts...)
        if !isone(α)
            res .*= α
        end
    else
        copyto!(res_backup, res)
        pullback!(f, (res,), prep, backend, x, (v,), contexts...)
        axpby!(β, res_backup, α, res)
    end
    return res
end

function build_A(
    implicit::ImplicitFunction{lazy},
    x::AbstractVector,
    y_or_yz,
    args...;
    suggested_backend,
    kwargs...,
) where {lazy}
    (; conditions, conditions_y_backend) = implicit
    y = output(y_or_yz)
    n, m = length(x), length(y)
    back_y = isnothing(conditions_y_backend) ? suggested_backend : conditions_y_backend
    cond_y = ConditionsY(conditions, kwargs)
    contexts = (Constant(x), map(Constant, rest(y_or_yz))..., map(Constant, args)...)
    if lazy
        prep = prepare_pushforward_same_point(cond_y, back_y, y, (zero(y),), contexts...)
        A = LinearOperator(
            eltype(y),
            m,
            m,
            false,
            false,
            PushforwardOperator!(cond_y, prep, back_y, y, contexts),
            typeof(y),
        )
    else
        J = jacobian(cond_y, back_y, y, contexts...)
        A = factorize(J)
    end
    return A
end

function build_Aᵀ(
    implicit::ImplicitFunction{lazy},
    x::AbstractVector,
    y_or_yz,
    args...;
    suggested_backend,
    kwargs...,
) where {lazy}
    (; conditions, conditions_y_backend) = implicit
    y = output(y_or_yz)
    n, m = length(x), length(y)
    back_y = isnothing(conditions_y_backend) ? suggested_backend : conditions_y_backend
    cond_y = ConditionsY(conditions, kwargs)
    contexts = (Constant(x), map(Constant, rest(y_or_yz))..., map(Constant, args)...)
    if lazy
        prep = prepare_pullback_same_point(cond_y, back_y, y, (zero(y),), contexts...)
        Aᵀ = LinearOperator(
            eltype(y),
            m,
            m,
            false,
            false,
            PullbackOperator!(cond_y, prep, back_y, y, contexts),
            typeof(y),
        )
    else
        Jᵀ = transpose(jacobian(cond_y, back_y, y, contexts...))
        Aᵀ = factorize(Jᵀ)
    end
    return Aᵀ
end

function build_B(
    implicit::ImplicitFunction{lazy},
    x::AbstractVector,
    y_or_yz,
    args...;
    suggested_backend,
    kwargs...,
) where {lazy}
    (; conditions, conditions_x_backend) = implicit
    y = output(y_or_yz)
    n, m = length(x), length(y)
    back_x = isnothing(conditions_x_backend) ? suggested_backend : conditions_x_backend
    cond_x = ConditionsX(conditions, kwargs)
    contexts = (Constant(y), map(Constant, rest(y_or_yz))..., map(Constant, args)...)
    if lazy
        prep = prepare_pushforward_same_point(cond_x, back_x, x, (zero(x),), contexts...)
        B = LinearOperator(
            eltype(y),
            m,
            n,
            false,
            false,
            PushforwardOperator!(cond_x, prep, back_x, x, contexts),
            typeof(x),
        )
    else
        B = transpose(jacobian(cond_x, back_x, x, contexts...))
    end
    return B
end

function build_Bᵀ(
    implicit::ImplicitFunction{lazy},
    x::AbstractVector,
    y_or_yz,
    args...;
    suggested_backend,
    kwargs...,
) where {lazy}
    (; conditions, conditions_x_backend) = implicit
    y = output(y_or_yz)
    n, m = length(x), length(y)
    back_x = isnothing(conditions_x_backend) ? suggested_backend : conditions_x_backend
    cond_x = ConditionsX(conditions, kwargs)
    contexts = (Constant(y), map(Constant, rest(y_or_yz))..., map(Constant, args)...)
    if lazy
        prep = prepare_pullback_same_point(cond_x, back_x, x, (zero(y),), contexts...)
        Bᵀ = LinearOperator(
            eltype(y),
            n,
            m,
            false,
            false,
            PullbackOperator!(cond_x, prep, back_x, x, contexts),
            typeof(x),
        )
    else
        Bᵀ = transpose(jacobian(cond_x, back_x, x, contexts...))
    end
    return Bᵀ
end
