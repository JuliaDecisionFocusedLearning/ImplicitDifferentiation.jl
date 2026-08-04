module ImplicitDifferentiationEnzymeExt

using ADTypes: AutoEnzyme
using EnzymeCore
using EnzymeCore: make_zero, set_runtime_activity
using EnzymeCore.EnzymeRules: EnzymeRules, AugmentedReturn, augmented_rule_return_type, needs_primal, width
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

function EnzymeRules.forward(
    config,
    implicit::Const{<:ImplicitFunction},
    ::Type{<:AnyDuplicated},
    x::AnyDuplicated,
    args::Vararg{Const,N},
) where {N}
    implicit = implicit.val

    dx = x.dval
    x = x.val
    args = ntuple(length(args)) do i
        return args[i].val
    end

    prep = ImplicitFunctionPreparation(eltype(x))
    (; conditions, linear_solver) = implicit

    y, z = implicit(x, args...)
    c = conditions(x, y, z, args...)

    y0 = zero(y)
    forward_backend = AutoEnzyme(;
        mode=set_runtime_activity(Forward), function_annotation=Const
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

    return if width(config) == 1
        dc = B(dx)
        dy = linear_solver(A, Aᵀ, -dc, y0)::typeof(y0)
        dz = make_zero(z)

        if needs_primal(config)
            return Duplicated((y, z), (dy, dz))
        else
            return dy, dz
        end
    else
        dc = map(B, dx)
        dy = map(dc) do dₖc
            return linear_solver(A, Aᵀ, -dₖc, y0)
        end

        df = ntuple(Val(width(config))) do i
            return (dy[i]::typeof(y0), dz::typeof(z))
        end

        if needs_primal(config)
            return BatchDuplicated((y, z), df)
        else
            # TODO: We need to heal the type instability from the linear solver here
            return df::NTuple{width(config),Tuple{typeof(y0),typeof(z)}}
        end
    end
end

function EnzymeRules.augmented_primal(
    config,
    implicit::Const{<:ImplicitFunction},
    RT::Type{<:AnyDuplicated},
    x::AnyDuplicated,
    args::Vararg{Const,N},
) where {N}
    @assert EnzymeRules.width(config) == 1
    implicit = implicit.val

    x = x.val
    args = ntuple(length(args)) do i
        return args[i].val
    end

    prep = ImplicitFunctionPreparation(eltype(x))
    (; conditions, linear_solver) = implicit

    y, z = implicit(x, args...)
    c = conditions(x, y, z, args...)
    c0 = zero(c)

    forward_backend = AutoEnzyme(; mode=Forward)
    reverse_backend = AutoEnzyme(; mode=Reverse)

    Aᵀ = build_Aᵀ(implicit, prep, x, y, z, c, args...; suggested_backend=reverse_backend)
    Bᵀ = build_Bᵀ(implicit, prep, x, y, z, c, args...; suggested_backend=reverse_backend)
    if linear_solver isa IterativeLeastSquaresSolver
        A = build_A(implicit, prep, x, y, z, c, args...; suggested_backend=forward_backend)
    else
        A = nothing
    end

    if EnzymeRules.needs_primal(config)
        primal = (y, z)
    else
        primal = nothing
    end

    dy = make_zero(y)
    if needs_shadow(config)
        shadow = (dy, make_zero(z))
    else
        shadow = nothing
    end

    tape = (; Aᵀ, Bᵀ, A, linear_solver, dy, c0)

    AR = augmented_rule_return_type(config, RT)

    return AR(primal, shadow, tape)
end

function EnzymeRules.reverse(
    _, ::Const{<:ImplicitFunction}, ::Type, tape, x::AnyDuplicated, ::Vararg{Const,N}
) where {N}
    dx = x.dval
    (; Aᵀ, Bᵀ, A, linear_solver, dy, c0) = tape

    dc = linear_solver(Aᵀ, A, -dy, c0)
    dx .+= Bᵀ(dc)

    return (nothing, nothing)
end

end # modul
