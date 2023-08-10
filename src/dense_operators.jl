#=

struct VecPartialConditions{whicharg,C,X,YZ,A,K}
    conditions::C
    x::X
    y_or_yz::YZ
    args::A
    kwargs::K

    function VecPartialConditions{whicharg}(
        conditions::C, x::X, y_or_yz::YZ, args::A, kwargs::K
    ) where {whicharg,C,X,YZ,A,K}
        return new{whicharg,C,X,YZ,A,K}(conditions, x, y_or_yz, args, kwargs)
    end
end

function (vpc::VecPartialConditions{1,C,X,<:AbstractArray})(_x_vec) where {C,X}
    @unpack conditions, x, y_or_yz, args, kwargs = vpc
    y = get_output(y_or_yz)
    _x = reshape(_x_vec, size(x))
    return vec(conditions(_x, y, args...; kwargs...))
end

function (vpc::VecPartialConditions{1,C,X,<:Tuple})(_x_vec) where {C,X}
    @unpack conditions, x, y_or_yz, args, kwargs = vpc
    y, z = get_output(y_or_yz), get_byproduct(y_or_yz)
    _x = reshape(_x_vec, size(x))
    return vec(conditions(_x, y, z, args...; kwargs...))
end

function (vpc::VecPartialConditions{2,C,X,<:AbstractArray})(_y_vec) where {C,X}
    @unpack conditions, x, y_or_yz, args, kwargs = vpc
    y = get_output(y_or_yz)
    _y = reshape(_y_vec, size(y))
    return vec(conditions(x, _y, args...; kwargs...))
end

function (vpc::VecPartialConditions{2,C,X,<:Tuple})(_y_vec) where {C,X}
    @unpack conditions, x, y_or_yz, args, kwargs = vpc
    y, z = get_output(y_or_yz), get_byproduct(y_or_yz)
    _y = reshape(_y_vec, size(y))
    return vec(conditions(x, _y, z, args...; kwargs...))
end

function conditions_partials(
    implicit::ImplicitFunction, x::AbstractArray, y::AbstractArray, args; kwargs
)
    conditions = implicit.conditions
    function conditions_yvec(y_vec)
        _y = reshape(y_vec, size(y))
        return vec(conditions(x, _y, args...; kwargs...))
    end
    function conditions_xvec(x_vec)
        _x = reshape(x_vec, size(x))
        return vec(conditions(_x, y, args...; kwargs...))
    end
    return conditions_yvec, conditions_xvec
end

function conditions_partials(
    implicit::ImplicitFunction, x::AbstractArray, yz::Tuple, args; kwargs
)
    conditions = implicit.conditions
    y, z = yz
    function conditions_yvec(y_vec)
        _y = reshape(y_vec, size(y))
        return vec(conditions(x, _y, z, args...; kwargs...))
    end
    function conditions_xvec(x_vec)
        _x = reshape(x_vec, size(x))
        return vec(conditions(_x, y, z, args...; kwargs...))
    end
    return conditions_yvec, conditions_xvec
end

## Forward

function conditions_forward_operators(
    ba::AbstractBackend,
    implicit::ImplicitFunction{F,C,DirectLinearSolver},
    x::AbstractArray,
    y_or_yz,
    args;
    kwargs,
) where {F,C}
    conditions_xvec = VecPartialConditions{1}(implicit.conditions, x, y_or_yz, args, kwargs)
    conditions_yvec = VecPartialConditions{2}(implicit.conditions, x, y_or_yz, args, kwargs)
    y = get_output(y_or_yz)
    A_vec = only(jacobian(ba, conditions_yvec, vec(y)))
    B_vec = only(jacobian(ba, conditions_xvec, vec(x)))
    return A_vec, B_vec
end

## Reverse

function conditions_reverse_operators(
    ba::AbstractBackend,
    implicit::ImplicitFunction{F,C,DirectLinearSolver},
    x::AbstractArray,
    y_or_yz,
    args;
    kwargs,
) where {F,C}
    conditions_yvec, conditions_xvec = conditions_partials(
        implicit, x, y_or_yz, args; kwargs
    )
    y = get_output(y_or_yz)
    A_vec = only(jacobian(ba, conditions_yvec, vec(y)))
    B_vec = only(jacobian(ba, conditions_xvec, vec(x)))
    return transpose(A_vec), transpose(B_vec)
end

=#
