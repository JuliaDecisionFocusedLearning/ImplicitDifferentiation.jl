module ImplicitDifferentiationEnzymeExt

using ADTypes: AutoEnzyme
using EnzymeCore
import EnzymeCore: EnzymeRules
using ImplicitDifferentiation:
    ImplicitFunction,
    ImplicitFunctionPreparation,
    IterativeLeastSquaresSolver,
    build_A,
    build_Aᵀ,
    build_B,
    build_Bᵀ

import .EnzymeRules: AugmentedReturn

const AnyDuplicated{T} = Union{Duplicated{T}, BatchDuplicated{T}, DuplicatedNoNeed{T}, BatchDuplicatedNoNeed{T}}

function EnzymeRules.forward(config, implicit::Const{<:ImplicitFunction}, ::Type{<:AnyDuplicated}, x::AnyDuplicated, args::Vararg{<:Const})
    implicit = implicit.val

    dx = x.dval
    x = x.val
    args = ntuple(length(args)) do i
        args[i].val
    end

    prep = ImplicitFunctionPreparation(eltype(x))
    (; conditions, linear_solver) = implicit

    y, z = implicit(x, args...)
    c = conditions(x, y, z, args...)

    y0 = zero(y)
    forward_backend = AutoEnzyme(mode = Forward)
    reverse_backend = AutoEnzyme(mode = Reverse)

    A = build_A(implicit, prep, x, y, z, c, args...; suggested_backend = forward_backend)
    B = build_B(implicit, prep, x, y, z, c, args...; suggested_backend = forward_backend)
    Aᵀ = if linear_solver isa IterativeLeastSquaresSolver
        build_Aᵀ(implicit, prep, x, y, z, c, args...; suggested_backend = reverse_backend)
    else
        nothing
    end

    return if EnzymeRules.width(config) == 1
        dc = B(dx)
        dy = linear_solver(A, Aᵀ, dc, y0)::typeof(y0)
        dz = nothing

        if EnzymeRules.needs_primal(config)
            return Duplicated((y, z), (dy, dz))
        else
            return dy, dz
        end
    else
        dc = map(B, dx)
        dy = map(dc) do dₖc
            linear_solver(A, Aᵀ, -dₖc, y0)
        end

        df = ntuple(Val(EnzymeRules.width(config))) do i
            (dy[i]::typeof(y0), nothing)
        end

        if EnzymeRules.needs_primal(config)
            return BatchDuplicated((y, z), df)
        else
            # TODO: We need to heal the type instability from the linear solver here
            # df::NTuple{EnzymeRules.width(config), Tuple{typeof(y0), Nothing}}
            return df::NTuple{EnzymeRules.width(config), Tuple{Vector{Float64}, Nothing}}
        end
    end
end

function EnzymeRules.augmented_primal(config, implicit::Const{<:ImplicitFunction}, RT::Type{<:AnyDuplicated}, x::AnyDuplicated, args::Vararg{<:Const})
    @assert EnzymeRules.width(config) == 1
    implicit = implicit.val

    x = x.val
    args = ntuple(length(args)) do i
        args[i].val
    end

    prep = ImplicitFunctionPreparation(eltype(x))
    (; conditions, linear_solver) = implicit

    y, z = implicit(x, args...)
    c = conditions(x, y, z, args...)
    c0 = zero(c)

    forward_backend = AutoEnzyme(mode = Forward)
    reverse_backend = AutoEnzyme(mode = Reverse)

    Aᵀ = build_Aᵀ(implicit, prep, x, y, z, c, args...; suggested_backend = reverse_backend)
    Bᵀ = build_Bᵀ(implicit, prep, x, y, z, c, args...; suggested_backend = reverse_backend)
    if linear_solver isa IterativeLeastSquaresSolver
        A = build_A(implicit, prep, x, y, z, c, args...; suggested_backend = forward_backend)
    else
        A = nothing
    end

    if EnzymeRules.needs_primal(config)
        primal = (y, z)
    else
        primal = nothing
    end

    dy = EnzymeCore.make_zero(y)
    if EnzymeRules.needs_shadow(config)
        shadow = (dy, EnzymeCore.make_zero(z))
    else
        shadow = nothing
    end

    tape = (; Aᵀ, Bᵀ, A, linear_solver, dy, c0)

    AR = EnzymeRules.augmented_rule_return_type(config, RT)

    return AR(primal, shadow, tape)
end

function EnzymeRules.reverse(_, ::Const{<:ImplicitFunction}, ::Type, tape, x::AnyDuplicated, ::Vararg{<:Const})
    dx = x.dval
    (; Aᵀ, Bᵀ, A, linear_solver, dy, c0) = tape

    dc = linear_solver(Aᵀ, A, -dy, c0)
    dx .+= Bᵀ(dc)

    return (nothing, nothing)
end

end # modul
