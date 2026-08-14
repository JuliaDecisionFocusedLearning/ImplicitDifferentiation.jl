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
    set_runtime_activity
using EnzymeCore.EnzymeRules:
    EnzymeRules, augmented_rule_return_type, needs_primal, needs_shadow, width
using ImplicitDifferentiation:
    ImplicitFunction,
    ImplicitFunctionPreparation,
    IterativeLeastSquaresSolver,
    build_A,
    build_Aᵀ,
    build_B,
    build_Bᵀ

const AnyDuplicated{T} = Union{
    Duplicated{T},BatchDuplicated{T},DuplicatedNoNeed{T},BatchDuplicatedNoNeed{T}
}

function _forward_setup(config, implicit::Const{<:ImplicitFunction}, x, args)
    implicit = implicit.val

    x = x.val
    args = map(a -> a.val, args)

    prep = ImplicitFunctionPreparation(eltype(x))
    (; conditions, linear_solver) = implicit

    y, z = implicit(x, args...)
    c = conditions(x, y, z, args...)

    y0 = zero(y)
    dz = make_zero(z)::typeof(z)

    forward_backend = AutoEnzyme(;
        mode=set_runtime_activity(Forward, config), function_annotation=Const
    )
    reverse_backend = AutoEnzyme(;
        mode=set_runtime_activity(Reverse), function_annotation=Const
    )

    A = build_A(implicit, prep, x, y, z, c, args...; suggested_backend=forward_backend)
    B = build_B(implicit, prep, x, y, z, c, args...; suggested_backend=forward_backend)
    Aᵀ = if linear_solver isa IterativeLeastSquaresSolver
        build_Aᵀ(implicit, prep, x, y, z, c, args...; suggested_backend=reverse_backend)
    else
        nothing
    end

    return (; y, z, y0, dz, A, B, Aᵀ, linear_solver)
end

function _forward_single(config, implicit, x, args)
    (; y, z, y0, dz, A, B, Aᵀ, linear_solver) = _forward_setup(config, implicit, x, args)
    dc = B(x.dval)
    dy = linear_solver(A, Aᵀ, -dc, y0)::typeof(y0)
    return y, z, dy, dz
end

function _forward_batch(config, implicit, x, args, ::Val{W}) where {W}
    (; y, z, y0, dz, A, B, Aᵀ, linear_solver) = _forward_setup(config, implicit, x, args)
    dc = map(B, x.dval)
    dy = map(dc) do dₖc
        return linear_solver(A, Aᵀ, -dₖc, y0)::typeof(y0)
    end
    df = ntuple(Val(W)) do i
        return (dy[i], dz)
    end::NTuple{W,Tuple{typeof(y0),typeof(z)}}
    return y, z, df
end

# Dispatching on the width (via the config type) and on whether the primal is needed (via
# the `RT` type) rather than branching at runtime on `width(config)`/`needs_primal(config)`
# keeps each method's return type concrete: a runtime branch between differently-shaped
# return values (e.g. `Duplicated` vs a bare tuple, or `BatchDuplicated` vs an `NTuple`)
# makes Enzyme infer a non-concrete return type for the whole function, which it rejects.
function EnzymeRules.forward(
    config::EnzymeRules.FwdConfigWidth{1},
    implicit::Const{<:ImplicitFunction},
    ::Type{<:Duplicated},
    x::Union{Duplicated,DuplicatedNoNeed},
    args::Vararg{Const,N},
) where {N}
    y, z, dy, dz = _forward_single(config, implicit, x, args)
    return Duplicated((y, z), (dy, dz))
end

function EnzymeRules.forward(
    config::EnzymeRules.FwdConfigWidth{1},
    implicit::Const{<:ImplicitFunction},
    ::Type{<:DuplicatedNoNeed},
    x::Union{Duplicated,DuplicatedNoNeed},
    args::Vararg{Const,N},
) where {N}
    y, z, dy, dz = _forward_single(config, implicit, x, args)
    return (dy, dz)
end

function EnzymeRules.forward(
    config::EnzymeRules.FwdConfigWidth{W},
    implicit::Const{<:ImplicitFunction},
    ::Type{<:BatchDuplicated},
    x::Union{BatchDuplicated,BatchDuplicatedNoNeed},
    args::Vararg{Const,N},
) where {W,N}
    y, z, df = _forward_batch(config, implicit, x, args, Val(W))
    return BatchDuplicated((y, z), df)
end

function EnzymeRules.forward(
    config::EnzymeRules.FwdConfigWidth{W},
    implicit::Const{<:ImplicitFunction},
    ::Type{<:BatchDuplicatedNoNeed},
    x::Union{BatchDuplicated,BatchDuplicatedNoNeed},
    args::Vararg{Const,N},
) where {W,N}
    y, z, df = _forward_batch(config, implicit, x, args, Val(W))
    return df
end

function EnzymeRules.augmented_primal(
    config,
    implicit::Const{<:ImplicitFunction},
    RT::Type{<:AnyDuplicated},
    x::AnyDuplicated,
    args::Vararg{Const,N},
) where {N}
    implicit = implicit.val

    x = x.val
    args = map(a -> a.val, args)

    prep = ImplicitFunctionPreparation(eltype(x))
    (; conditions, linear_solver) = implicit

    y, z = implicit(x, args...)
    c = conditions(x, y, z, args...)
    c0 = zero(c)

    forward_backend = AutoEnzyme(; mode=set_runtime_activity(Forward))
    reverse_backend = AutoEnzyme(; mode=set_runtime_activity(Reverse))

    Aᵀ = build_Aᵀ(implicit, prep, x, y, z, c, args...; suggested_backend=reverse_backend)
    Bᵀ = build_Bᵀ(implicit, prep, x, y, z, c, args...; suggested_backend=reverse_backend)
    if linear_solver isa IterativeLeastSquaresSolver
        A = build_A(implicit, prep, x, y, z, c, args...; suggested_backend=forward_backend)
    else
        A = nothing
    end

    if needs_primal(config)
        primal = (y, z)
    else
        primal = nothing
    end

    W = width(config)
    dy = W == 1 ? make_zero(y) : ntuple(_ -> make_zero(y), Val(W))
    dz = W == 1 ? make_zero(z) : ntuple(_ -> make_zero(z), Val(W))
    if needs_shadow(config)
        shadow = W == 1 ? (dy, dz) : ntuple(i -> (dy[i], dz[i]), Val(W))
    else
        shadow = nothing
    end

    tape = (; Aᵀ, Bᵀ, A, linear_solver, dy, c0)

    AR = augmented_rule_return_type(config, RT)

    return AR(primal, shadow, tape)
end

function EnzymeRules.reverse(
    config, ::Const{<:ImplicitFunction}, ::Type, tape, x::AnyDuplicated, ::Vararg{Const,N}
) where {N}
    dx = x.dval
    (; Aᵀ, Bᵀ, A, linear_solver, dy, c0) = tape

    if width(config) == 1
        dc = linear_solver(Aᵀ, A, -dy, c0)
        dx .+= Bᵀ(dc)
    else
        for i in eachindex(dy)
            dc = linear_solver(Aᵀ, A, -dy[i], c0)
            dx[i] .+= Bᵀ(dc)
        end
    end

    return (nothing, ntuple(_ -> nothing, Val(N))...)
end

end # module
