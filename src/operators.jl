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
    y = get_output(y_or_yz)
    if y_or_yz isa Tuple
        z = get_byproduct(y_or_yz)
        return ConditionsXByproduct(conditions, y, z, args, kwargs)
    else
        return ConditionsXNoByproduct(conditions, y, args, kwargs)
    end
end

function ConditionsY(conditions, x, y_or_yz, args...; kwargs...)
    if y_or_yz isa Tuple
        z = get_byproduct(y_or_yz)
        return ConditionsYByproduct(conditions, x, z, args, kwargs)
    else
        return ConditionsYNoByproduct(conditions, x, args, kwargs)
    end
end

## Lazy operators

function build_A(
    implicit::ImplicitFunction,
    x::AbstractVector,
    y_or_yz,
    args...;
    suggested_backend,
    kwargs...,
)
    (; conditions, linear_solver, conditions_y_backend) = implicit
    y = get_output(y_or_yz)
    n, m = length(x), length(y)
    back_y = isnothing(conditions_y_backend) ? suggested_backend : conditions_y_backend
    cond_y = ConditionsY(conditions, x, y_or_yz, args...; kwargs...)
    if linear_solver isa typeof(\)
        A = factorize(jacobian(cond_y, back_y, y))
    else
        extras = prepare_pushforward(cond_y, back_y, y)
        A = LinearOperator(
            eltype(y),
            m,
            m,
            false,
            false,
            (res, v) -> res .= pushforward!!(cond_y, res, back_y, y, v, extras),
            typeof(y),
        )
    end
    return A
end

function build_Aᵀ(
    implicit::ImplicitFunction,
    x::AbstractVector,
    y_or_yz,
    args...;
    suggested_backend,
    kwargs...,
)
    (; conditions, linear_solver, conditions_y_backend) = implicit
    y = get_output(y_or_yz)
    n, m = length(x), length(y)
    back_y = isnothing(conditions_y_backend) ? suggested_backend : conditions_y_backend
    cond_y = ConditionsY(conditions, x, y_or_yz, args...; kwargs...)
    if linear_solver isa typeof(\)
        Aᵀ = factorize(transpose(jacobian(cond_y, back_y, y)))
    else
        extras = prepare_pullback(cond_y, back_y, y)
        _, pullbackfunc!! = value_and_pullback!!_split(cond_y, back_y, y, extras)
        Aᵀ = LinearOperator(
            eltype(y),
            m,
            m,
            false,
            false,
            (res, v) -> res .= pullbackfunc!!(res, v),
            typeof(y),
        )
    end
    return Aᵀ
end

function build_B(
    implicit::ImplicitFunction,
    x::AbstractVector,
    y_or_yz,
    args...;
    suggested_backend,
    kwargs...,
)
    (; conditions, linear_solver, conditions_x_backend) = implicit
    y = get_output(y_or_yz)
    n, m = length(x), length(y)
    back_x = isnothing(conditions_x_backend) ? suggested_backend : conditions_x_backend
    cond_x = ConditionsX(conditions, x, y_or_yz, args...; kwargs...)
    if linear_solver isa typeof(\)
        B = factorize(transpose(jacobian(cond_x, back_x, x)))
    else
        extras = prepare_pushforward(cond_x, back_x, x)
        B = LinearOperator(
            eltype(y),
            m,
            n,
            false,
            false,
            (res, v) -> res .= pushforward!!(cond_x, res, back_x, x, v, extras),
            typeof(x),
        )
    end
    return B
end

function build_Bᵀ(
    implicit::ImplicitFunction,
    x::AbstractVector,
    y_or_yz,
    args...;
    suggested_backend,
    kwargs...,
)
    (; conditions, linear_solver, conditions_x_backend) = implicit
    y = get_output(y_or_yz)
    n, m = length(x), length(y)
    back_x = isnothing(conditions_x_backend) ? suggested_backend : conditions_x_backend
    cond_x = ConditionsX(conditions, x, y_or_yz, args...; kwargs...)
    if linear_solver isa typeof(\)
        Bᵀ = factorize(transpose(jacobian(cond_x, back_x, x)))
    else
        extras = prepare_pullback(cond_x, back_x, x)
        _, pullbackfunc!! = value_and_pullback!!_split(cond_x, back_x, x, extras)
        Bᵀ = LinearOperator(
            eltype(y),
            n,
            m,
            false,
            false,
            (res, v) -> res .= pullbackfunc!!(res, v),
            typeof(x),
        )
    end
    return Bᵀ
end
