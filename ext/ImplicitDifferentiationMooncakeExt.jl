module ImplicitDifferentiationMooncakeExt

using ADTypes: AutoMooncake, AutoMooncakeForward
using ImplicitDifferentiation:
    ImplicitFunction,
    ImplicitFunctionPreparation,
    IterativeLeastSquaresSolver,
    build_A,
    build_Aᵀ,
    build_B,
    build_Bᵀ
using Mooncake:
    Mooncake,
    @is_primitive,
    AsPrimal,
    CoDual,
    DefaultCtx,
    Dual,
    FriendlyTangentCache,
    NoRData,
    NoTangent,
    fdata,
    increment!!,
    prepare_derivative_cache,
    prepare_pullback_cache,
    primal,
    primal_to_tangent!!,
    rdata,
    tangent,
    tangent_to_friendly!!,
    tangent_type,
    value_and_derivative!!,
    value_and_pullback!!,
    zero_fcodual,
    zero_rdata,
    zero_tangent

# The conditions are differentiated with Mooncake too, unless `backends` says otherwise.
const FORWARD_BACKEND = AutoMooncakeForward(; config=nothing)
const REVERSE_BACKEND = AutoMooncake(; config=nothing)

@is_primitive DefaultCtx Tuple{<:ImplicitFunction,<:AbstractArray,Vararg}
@is_primitive DefaultCtx Tuple{
    <:ImplicitFunction,<:ImplicitFunctionPreparation,<:AbstractArray,Vararg
}

## Conversions between Mooncake tangents and arrays shaped like the primal

function tangent_to_array(x::AbstractArray, t)
    if t isa typeof(x)
        return t
    else
        dest = FriendlyTangentCache{AsPrimal}(copy(x))
        return tangent_to_friendly!!(dest, x, t, IdDict{Any,Any}())
    end
end

function array_to_tangent(x::AbstractArray, a::AbstractArray)
    if a isa tangent_type(typeof(x))
        return a
    else
        return primal_to_tangent!!(zero_tangent(x), a)
    end
end

## Dependence of the conditions on everything except `x` and `y`

"""
    ConditionsAt(x, y, z)

The conditions as a function of the `ImplicitFunction` (whose captured data they may
read) and of the positional arguments beyond `x`, at fixed `x`, `y` and `z`.
"""
struct ConditionsAt{X,Y,Z}
    x::X
    y::Y
    z::Z
end

function (f::ConditionsAt)(implicit::ImplicitFunction, args::Vararg{Any,N}) where {N}
    return implicit.conditions(f.x, f.y, f.z, args...)
end

function has_other_tangents(implicit::ImplicitFunction, args::Tuple)
    return tangent_type(typeof((implicit, args...))) !== NoTangent
end

## Forward mode

function Mooncake.frule!!(
    implicit::Dual{<:ImplicitFunction}, x::Dual{<:AbstractArray}, args::Vararg{Dual,N}
) where {N}
    prep = ImplicitFunctionPreparation(eltype(primal(x)))
    return implicit_frule(implicit, prep, x, args...)
end

function Mooncake.frule!!(
    implicit::Dual{<:ImplicitFunction},
    prep::Dual{<:ImplicitFunctionPreparation},
    x::Dual{<:AbstractArray},
    args::Vararg{Dual,N},
) where {N}
    return implicit_frule(implicit, primal(prep), x, args...)
end

function implicit_frule(implicit_dual::Dual, prep, x::Dual, args::Vararg{Dual,N}) where {N}
    implicit = primal(implicit_dual)
    (; conditions, linear_solver) = implicit
    x0 = primal(x)
    args0 = map(primal, args)
    y, z = implicit(prep, x0, args0...)
    c = conditions(x0, y, z, args0...)
    A = build_A(implicit, prep, x0, y, z, c, args0...; suggested_backend=FORWARD_BACKEND)
    B = build_B(implicit, prep, x0, y, z, c, args0...; suggested_backend=FORWARD_BACKEND)
    Aᵀ = if linear_solver isa IterativeLeastSquaresSolver
        build_Aᵀ(implicit, prep, x0, y, z, c, args0...; suggested_backend=REVERSE_BACKEND)
    else
        nothing
    end
    dc = B(tangent_to_array(x0, tangent(x)))
    if has_other_tangents(implicit, args0)
        f = ConditionsAt(x0, y, z)
        cache = prepare_derivative_cache(f, implicit, args0...)
        output = value_and_derivative!!(
            cache, Dual(f, zero_tangent(f)), implicit_dual, args...
        )
        dc = dc .+ tangent_to_array(c, tangent(output))
    end
    dy = linear_solver(A, Aᵀ, -dc, zero(y))
    ty = array_to_tangent(y, copyto!(similar(y), dy))
    return Dual((y, z), (ty, zero_tangent(z)))
end

## Reverse mode

struct ImplicitPullback{P,TA,TB,TA2,S,Y,FY,Z,X,FX,C,I,FI,R,FR}
    prep_rdata::P
    Aᵀ::TA
    Bᵀ::TB
    A::TA2
    linear_solver::S
    y::Y
    fy::FY
    z::Z
    x::X
    fx::FX
    c::C
    implicit::I
    fimplicit::FI
    args::R
    fargs::FR
end

function (pb::ImplicitPullback)(dout)
    (; prep_rdata, Aᵀ, Bᵀ, A, linear_solver, y, fy, x, fx, c, implicit, fimplicit) = pb
    (; z, args, fargs) = pb
    ry = if dout isa NoRData
        NoRData()
    else
        first(dout)
    end
    dy = tangent_to_array(y, tangent(fy, ry))
    dc = linear_solver(Aᵀ, A, -dy, zero(c))
    tx = array_to_tangent(x, copyto!(similar(x), Bᵀ(dc)))
    increment!!(fx, fdata(tx))
    if has_other_tangents(implicit, args)
        f = ConditionsAt(x, y, z)
        cache = prepare_pullback_cache(f, implicit, args...)
        tc = array_to_tangent(c, copyto!(similar(c), dc))
        _, (_, timplicit, targs...) = value_and_pullback!!(cache, tc, f, implicit, args...)
        increment!!(fimplicit, fdata(timplicit))
        foreach((farg, targ) -> increment!!(farg, fdata(targ)), fargs, targs)
        return rdata(timplicit), prep_rdata..., rdata(tx), map(rdata, targs)...
    else
        return zero_rdata(implicit), prep_rdata..., rdata(tx), map(zero_rdata, args)...
    end
end

function Mooncake.rrule!!(
    implicit::CoDual{<:ImplicitFunction}, x::CoDual{<:AbstractArray}, args::Vararg{CoDual,N}
) where {N}
    prep = ImplicitFunctionPreparation(eltype(primal(x)))
    return implicit_rrule((), implicit, prep, x, args...)
end

function Mooncake.rrule!!(
    implicit::CoDual{<:ImplicitFunction},
    prep::CoDual{<:ImplicitFunctionPreparation},
    x::CoDual{<:AbstractArray},
    args::Vararg{CoDual,N},
) where {N}
    return implicit_rrule((zero_rdata(primal(prep)),), implicit, primal(prep), x, args...)
end

function implicit_rrule(
    prep_rdata::Tuple, implicit_codual::CoDual, prep, x::CoDual, args::Vararg{CoDual,N}
) where {N}
    implicit = primal(implicit_codual)
    (; conditions, linear_solver) = implicit
    x0 = primal(x)
    args0 = map(primal, args)
    y, z = implicit(prep, x0, args0...)
    c = conditions(x0, y, z, args0...)
    Aᵀ = build_Aᵀ(implicit, prep, x0, y, z, c, args0...; suggested_backend=REVERSE_BACKEND)
    Bᵀ = build_Bᵀ(implicit, prep, x0, y, z, c, args0...; suggested_backend=REVERSE_BACKEND)
    A = if linear_solver isa IterativeLeastSquaresSolver
        build_A(implicit, prep, x0, y, z, c, args0...; suggested_backend=FORWARD_BACKEND)
    else
        nothing
    end
    output = zero_fcodual((y, z))
    pullback = ImplicitPullback(
        prep_rdata,
        Aᵀ,
        Bᵀ,
        A,
        linear_solver,
        y,
        first(tangent(output)),
        z,
        x0,
        tangent(x),
        c,
        implicit,
        tangent(implicit_codual),
        args0,
        map(tangent, args),
    )
    return output, pullback
end

end # module
