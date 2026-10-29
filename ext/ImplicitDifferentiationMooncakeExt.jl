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
    CoDual,
    DefaultCtx,
    Dual,
    primal,
    tangent,
    zero_fcodual,
    zero_rdata,
    zero_tangent

# The conditions are differentiated with Mooncake too, unless `backends` says otherwise.
const FORWARD_BACKEND = AutoMooncakeForward(; config=nothing)
const REVERSE_BACKEND = AutoMooncake(; config=nothing)

# Only `x` is differentiated: the `ImplicitFunction` and `args` are constant.
@is_primitive DefaultCtx Tuple{<:ImplicitFunction,<:Array{<:Base.IEEEFloat},Vararg}

## Forward mode

function Mooncake.frule!!(
    implicit_dual::Dual{<:ImplicitFunction},
    x::Dual{<:Array{<:Base.IEEEFloat}},
    args::Vararg{Dual,N},
) where {N}
    implicit = primal(implicit_dual)
    (; conditions, linear_solver) = implicit
    x0 = primal(x)
    args0 = map(primal, args)
    y, z = implicit(x0, args0...)
    c = conditions(x0, y, z, args0...)
    prep = ImplicitFunctionPreparation(eltype(x0))
    A = build_A(implicit, prep, x0, y, z, c, args0...; suggested_backend=FORWARD_BACKEND)
    B = build_B(implicit, prep, x0, y, z, c, args0...; suggested_backend=FORWARD_BACKEND)
    Aᵀ = if linear_solver isa IterativeLeastSquaresSolver
        build_Aᵀ(implicit, prep, x0, y, z, c, args0...; suggested_backend=REVERSE_BACKEND)
    else
        nothing
    end
    dy = linear_solver(A, Aᵀ, -B(tangent(x)), zero(y))
    return Dual((y, z), (copyto!(similar(y), dy), zero_tangent(z)))
end

## Reverse mode

struct ImplicitPullback{TA,TB,TA2,S,FY,FX,C,R}
    Aᵀ::TA
    Bᵀ::TB
    A::TA2
    linear_solver::S
    fy::FY
    fx::FX
    c::C
    rdatas::R
end

function (pb::ImplicitPullback)(_)
    (; Aᵀ, Bᵀ, A, linear_solver, fy, fx, c, rdatas) = pb
    dc = linear_solver(Aᵀ, A, -fy, zero(c))
    fx .+= reshape(Bᵀ(dc), size(fx))
    return rdatas
end

function Mooncake.rrule!!(
    implicit_codual::CoDual{<:ImplicitFunction},
    x::CoDual{<:Array{<:Base.IEEEFloat}},
    args::Vararg{CoDual,N},
) where {N}
    implicit = primal(implicit_codual)
    (; conditions, linear_solver) = implicit
    x0 = primal(x)
    args0 = map(primal, args)
    y, z = implicit(x0, args0...)
    c = conditions(x0, y, z, args0...)
    prep = ImplicitFunctionPreparation(eltype(x0))
    Aᵀ = build_Aᵀ(implicit, prep, x0, y, z, c, args0...; suggested_backend=REVERSE_BACKEND)
    Bᵀ = build_Bᵀ(implicit, prep, x0, y, z, c, args0...; suggested_backend=REVERSE_BACKEND)
    A = if linear_solver isa IterativeLeastSquaresSolver
        build_A(implicit, prep, x0, y, z, c, args0...; suggested_backend=FORWARD_BACKEND)
    else
        nothing
    end
    output = zero_fcodual((y, z))
    rdatas = (zero_rdata(implicit), zero_rdata(x0), map(zero_rdata, args0)...)
    pullback = ImplicitPullback(
        Aᵀ, Bᵀ, A, linear_solver, first(tangent(output)), tangent(x), c, rdatas
    )
    return output, pullback
end

end # module
