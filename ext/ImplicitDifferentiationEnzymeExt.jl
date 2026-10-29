module ImplicitDifferentiationEnzymeExt

using ADTypes: AutoEnzyme
using EnzymeCore:
    BatchDuplicated,
    BatchDuplicatedNoNeed,
    Const,
    Duplicated,
    DuplicatedNoNeed,
    Forward,
    Reverse,
    make_zero,
    set_runtime_activity,
    set_strong_zero
using EnzymeCore.EnzymeRules: EnzymeRules, FwdConfig, RevConfig, overwritten, strong_zero
using ImplicitDifferentiation:
    ImplicitFunction,
    ImplicitFunctionPreparation,
    IterativeLeastSquaresSolver,
    build_A,
    build_Aᵀ,
    build_B,
    build_Bᵀ

const AnyDuplicated = Union{
    Duplicated,BatchDuplicated,DuplicatedNoNeed,BatchDuplicatedNoNeed
}

# Enzyme requires shadows to have the same (inferrable) type as their primal, but linear
# solvers may return another array type (e.g. `A \ b` gives a `Vector` for a
# `ComponentVector`) or be type-unstable (e.g. when the inner Jacobian is).
function shadow_like(y::Y, dy) where {Y}
    return (dy isa Y ? dy : copyto!(make_zero(y), dy))::Y
end

# Tangents of `x` as a tuple, whatever the batch width.
tangents(x::Union{Duplicated,DuplicatedNoNeed}) = (x.dval,)
tangents(x::Union{BatchDuplicated,BatchDuplicatedNoNeed}) = x.dval

# The inner backends differentiating the conditions inherit the strong zero setting of the
# outer config (a type parameter, so the result is inferrable). Runtime activity is always
# enabled, because the `Constant` contexts of DifferentiationInterface often require it:
# without it, even outer modes without runtime activity would fail.
function inner_backends(config::Union{FwdConfig,RevConfig})
    sz = strong_zero(config)
    forward_mode = set_runtime_activity(sz ? set_strong_zero(Forward) : Forward)
    reverse_mode = set_runtime_activity(sz ? set_strong_zero(Reverse) : Reverse)
    return (
        AutoEnzyme(; mode=forward_mode, function_annotation=Const),
        AutoEnzyme(; mode=reverse_mode, function_annotation=Const),
    )
end

## Forward

# The expected return type is given by `EnzymeRules.forward_rule_return_type`. Dispatching
# on the config type parameters rather than branching at runtime keeps it concrete.
# These parameters are documented in the docstring of `FwdConfig`: they are
# `FwdConfig{NeedsPrimal,NeedsShadow,Width,RuntimeActivity,StrongZero}`.
function forward_return(::FwdConfig{true,true,1}, primal, shadows)
    return Duplicated(primal, only(shadows))
end
forward_return(::FwdConfig{true,true}, primal, shadows) = BatchDuplicated(primal, shadows)
forward_return(::FwdConfig{false,true,1}, primal, shadows) = only(shadows)
forward_return(::FwdConfig{false,true}, primal, shadows) = shadows
forward_return(::FwdConfig{true,false}, primal, shadows) = primal
forward_return(::FwdConfig{false,false}, primal, shadows) = nothing

# One zero shadow of the output `(y, z)` per batch direction.
zero_shadows(y, z, ::Val{W}) where {W} = ntuple(_ -> (make_zero(y), make_zero(z)), Val(W))

forward_shadows(::FwdConfig{<:Any,false}, implicit, x, y, z, args) = nothing

# Enzyme may call the rule with a constant `x` (e.g. when the output is unused).
function forward_shadows(
    ::FwdConfig{<:Any,true,W}, implicit, x::Const, y, z, args
) where {W}
    return zero_shadows(y, z, Val(W))
end

function forward_shadows(
    config::FwdConfig{<:Any,true}, implicit, x::AnyDuplicated, y, z, args
)
    (; conditions, linear_solver) = implicit
    c = conditions(x.val, y, z, args...)
    y0 = zero(y)
    forward_backend, reverse_backend = inner_backends(config)
    prep = ImplicitFunctionPreparation(eltype(x.val))
    A = build_A(implicit, prep, x.val, y, z, c, args...; suggested_backend=forward_backend)
    B = build_B(implicit, prep, x.val, y, z, c, args...; suggested_backend=forward_backend)
    Aᵀ = if linear_solver isa IterativeLeastSquaresSolver
        build_Aᵀ(implicit, prep, x.val, y, z, c, args...; suggested_backend=reverse_backend)
    else
        nothing
    end
    return map(tangents(x)) do dₖx
        dₖy = shadow_like(y, linear_solver(A, Aᵀ, -B(dₖx), y0))
        return (dₖy, make_zero(z))
    end
end

function EnzymeRules.forward(
    config::FwdConfig,
    implicit::Const{<:ImplicitFunction},
    ::Type{<:Union{Const,AnyDuplicated}},
    x::Union{Const,AnyDuplicated},
    args::Vararg{Const,N},
) where {N}
    implicit = implicit.val
    args = map(a -> a.val, args)
    y, z = implicit(x.val, args...)
    shadows = forward_shadows(config, implicit, x, y, z, args)
    return forward_return(config, (y, z), shadows)
end

## Reverse

function EnzymeRules.augmented_primal(
    config::RevConfig,
    implicit::Const{<:ImplicitFunction},
    ::Type{<:Union{Const,AnyDuplicated}},
    x::Union{Const,AnyDuplicated},
    args::Vararg{Const,N},
) where {N}
    implicit = implicit.val
    args = map(a -> a.val, args)
    y, z = implicit(x.val, args...)
    primal = EnzymeRules.needs_primal(config) ? (y, z) : nothing
    shadow, tape = augmented_shadow_and_tape(config, implicit, x, y, z, args)
    return EnzymeRules.AugmentedReturn(primal, shadow, tape)
end

function augmented_shadow_and_tape(::RevConfig{<:Any,false}, implicit, x, y, z, args)
    return nothing, nothing
end

# With a constant `x`, there is nothing to pull back.
function augmented_shadow_and_tape(
    config::RevConfig{<:Any,true,W}, implicit, x::Const, y, z, args
) where {W}
    return batch_shadow(config, zero_shadows(y, z, Val(W))), nothing
end

# Type parameters of `RevConfig`, as documented in its docstring:
# `RevConfig{NeedsPrimal,NeedsShadow,Width,Overwritten,RuntimeActivity,StrongZero}`.
function augmented_shadow_and_tape(
    config::RevConfig{<:Any,true,W}, implicit, x::AnyDuplicated, y, z, args
) where {W}
    shadows = zero_shadows(y, z, Val(W))
    # the pullback captures `x` and `y`, which may be mutated before the reverse pass
    x_tape = overwritten(config)[2] ? copy(x.val) : x.val
    pullback = build_pullback(config, implicit, x_tape, copy(y), z, args)
    return batch_shadow(config, shadows), (; pullback, shadows)
end

batch_shadow(::RevConfig{<:Any,<:Any,1}, shadows) = only(shadows)
batch_shadow(::RevConfig, shadows) = shadows

function build_pullback(config, implicit, x, y, z, args)
    (; conditions, linear_solver) = implicit
    c = conditions(x, y, z, args...)
    c0 = zero(c)
    forward_backend, reverse_backend = inner_backends(config)
    prep = ImplicitFunctionPreparation(eltype(x))
    Aᵀ = build_Aᵀ(implicit, prep, x, y, z, c, args...; suggested_backend=reverse_backend)
    Bᵀ = build_Bᵀ(implicit, prep, x, y, z, c, args...; suggested_backend=reverse_backend)
    A = if linear_solver isa IterativeLeastSquaresSolver
        build_A(implicit, prep, x, y, z, c, args...; suggested_backend=forward_backend)
    else
        nothing
    end
    return dy -> Bᵀ(linear_solver(Aᵀ, A, -dy, c0))
end

function EnzymeRules.reverse(
    config::RevConfig,
    ::Const{<:ImplicitFunction},
    ::Type{<:Union{Const,AnyDuplicated}},
    tape,
    x::Union{Const,AnyDuplicated},
    ::Vararg{Const,N},
) where {N}
    if !isnothing(tape)
        (; pullback, shadows) = tape
        # the byproduct `z` is treated as a constant, so its shadow `dₖz` is ignored
        # (unlike ChainRules, Enzyme has no way to mark it as not implemented)
        foreach(tangents(x), shadows) do dₖx, (dₖy, _)
            dₖx .+= pullback(dₖy)
            # the output shadow has been consumed, reset it to zero
            fill!(dₖy, zero(eltype(dₖy)))
            return nothing
        end
    end
    return (nothing, ntuple(Returns(nothing), Val(N))...)
end

end # module
