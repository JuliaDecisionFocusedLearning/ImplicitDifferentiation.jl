## Partial conditions

struct ConditionsXNoByproduct{C,Y,A,K}
    conditions::C
    y::Y
    args::A
    kwargs::K
end

function (conditions_x_nobyproduct::ConditionsXNoByproduct)(x::AbstractVector)
    (; conditions, y, args, kwargs) = conditions_x_nobyproduct
    return conditions(x, y, args...; kwargs...)
end

struct ConditionsYNoByproduct{C,X,A,K}
    conditions::C
    x::X
    args::A
    kwargs::K
end

function (conditions_y_nobyproduct::ConditionsYNoByproduct)(y::AbstractVector)
    (; conditions, x, args, kwargs) = conditions_y_nobyproduct
    return conditions(x, y, args...; kwargs...)
end

struct ConditionsXByproduct{C,Y,Z,A,K}
    conditions::C
    y::Y
    z::Z
    args::A
    kwargs::K
end

function (conditions_x_byproduct::ConditionsXByproduct)(x::AbstractVector)
    (; conditions, y, z, args, kwargs) = conditions_x_byproduct
    return conditions(x, y, z, args...; kwargs...)
end

struct ConditionsYByproduct{C,X,Z,A,K}
    conditions::C
    x::X
    z::Z
    args::A
    kwargs::K
end

function (conditions_y_byproduct::ConditionsYByproduct)(y::AbstractVector)
    (; conditions, x, z, args, kwargs) = conditions_y_byproduct
    return conditions(x, y, z, args...; kwargs...)
end

function ConditionsX(conditions, x, y_or_yz, args...; kwargs...)
    y = output(y_or_yz)
    if y_or_yz isa Tuple
        z = byproduct(y_or_yz)
        return ConditionsXByproduct(conditions, y, z, args, kwargs)
    else
        return ConditionsXNoByproduct(conditions, y, args, kwargs)
    end
end

function ConditionsY(conditions, x, y_or_yz, args...; kwargs...)
    if y_or_yz isa Tuple
        z = byproduct(y_or_yz)
        return ConditionsYByproduct(conditions, x, z, args, kwargs)
    else
        return ConditionsYNoByproduct(conditions, x, args, kwargs)
    end
end

## Lazy operators

struct PushforwardOperator!{F,B,X,E,R}
    f::F
    backend::B
    x::X
    extras::E
    res_backup::R
end

function (po::PushforwardOperator!)(res, v, α, β)
    if iszero(β)
        pushforward!(po.f, res, po.backend, po.x, v, po.extras)
        res .= α .* res
    else
        po.res_backup .= res
        pushforward!(po.f, res, po.backend, po.x, v, po.extras)
        res .= α .* res .+ β .* po.res_backup
    end
    return res
end

struct PullbackOperator!{PB,R}
    pullbackfunc!::PB
    res_backup::R
end

function (po::PullbackOperator!)(res, v, α, β)
    if iszero(β)
        po.pullbackfunc!(res, v)
    else
        po.res_backup .= res
        po.pullbackfunc!(res, v)
        res .= α .* res .+ β .+ po.res_backup
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
    (; conditions, linear_solver, conditions_y_backend) = implicit
    y = output(y_or_yz)
    n, m = length(x), length(y)
    back_y = isnothing(conditions_y_backend) ? suggested_backend : conditions_y_backend
    cond_y = ConditionsY(conditions, x, y_or_yz, args...; kwargs...)
    if lazy
        extras = prepare_pushforward(cond_y, back_y, y, similar(y))
        A = LinearOperator(
            eltype(y),
            m,
            m,
            false,
            false,
            PushforwardOperator!(cond_y, back_y, y, extras, similar(y)),
            typeof(y),
        )
    else
        J = jacobian(cond_y, back_y, y)
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
    (; conditions, linear_solver, conditions_y_backend) = implicit
    y = output(y_or_yz)
    n, m = length(x), length(y)
    back_y = isnothing(conditions_y_backend) ? suggested_backend : conditions_y_backend
    cond_y = ConditionsY(conditions, x, y_or_yz, args...; kwargs...)
    if lazy
        extras = prepare_pullback(cond_y, back_y, y, similar(y))
        _, pullbackfunc! = value_and_pullback!_split(cond_y, back_y, y, extras)
        Aᵀ = LinearOperator(
            eltype(y),
            m,
            m,
            false,
            false,
            PullbackOperator!(pullbackfunc!, similar(y)),
            typeof(y),
        )
    else
        Jᵀ = transpose(jacobian(cond_y, back_y, y))
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
    (; conditions, linear_solver, conditions_x_backend) = implicit
    y = output(y_or_yz)
    n, m = length(x), length(y)
    back_x = isnothing(conditions_x_backend) ? suggested_backend : conditions_x_backend
    cond_x = ConditionsX(conditions, x, y_or_yz, args...; kwargs...)
    if lazy
        extras = prepare_pushforward(cond_x, back_x, x, similar(x))
        B = LinearOperator(
            eltype(y),
            m,
            n,
            false,
            false,
            PushforwardOperator!(cond_x, back_x, x, extras, similar(y)),
            typeof(x),
        )
    else
        B = transpose(jacobian(cond_x, back_x, x))
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
    (; conditions, linear_solver, conditions_x_backend) = implicit
    y = output(y_or_yz)
    n, m = length(x), length(y)
    back_x = isnothing(conditions_x_backend) ? suggested_backend : conditions_x_backend
    cond_x = ConditionsX(conditions, x, y_or_yz, args...; kwargs...)
    if lazy
        extras = prepare_pullback(cond_x, back_x, x, similar(y))
        _, pullbackfunc! = value_and_pullback!_split(cond_x, back_x, x, extras)
        Bᵀ = LinearOperator(
            eltype(y),
            n,
            m,
            false,
            false,
            PullbackOperator!(pullbackfunc!, similar(y)),
            typeof(x),
        )
    else
        Bᵀ = transpose(jacobian(cond_x, back_x, x))
    end
    return Bᵀ
end
