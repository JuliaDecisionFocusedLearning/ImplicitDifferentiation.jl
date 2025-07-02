"""
    ImplicitFunctionPreparation

# Fields

- `prep_A`: preparation for `A` (derivative of conditions with respect to `y`) in forward mode
- `prep_Aᵀ`: preparation for `A` (derivative of conditions with respect to `y`) in reverse mode
- `prep_B`: preparation for `B` (derivative of conditions with respect to `x`) in forward mode
- `prep_Bᵀ`: preparation for `B` (derivative of conditions with respect to `x`) in reverse mode
"""
struct ImplicitFunctionPreparation{R<:Real,PA,PAT,PB,PBT}
    _R::Type{R}
    prep_A::PA
    prep_Aᵀ::PAT
    prep_B::PB
    prep_Bᵀ::PBT
end

function ImplicitFunctionPreparation(::Type{R}) where {R<:Real}
    return ImplicitFunctionPreparation(R, nothing, nothing, nothing, nothing)
end

"""
    prepare_implicit(
        mode::ADTypes.AbstractMode,
        implicit::ImplicitFunction,
        x_prep,
        args_prep...;
        strict=Val(true)
    )

Uses the preparation mechanism from [DifferentiationInterface.jl](https://github.com/JuliaDiff/DifferentiationInterface.jl) to speed up subsequent calls to `implicit(x, args...)` where `(x, args...)` are similar to `(x_prep, args_prep...)`.
"""
function prepare_implicit(
    mode::AbstractMode, implicit::ImplicitFunction, x, args::Vararg{Any,N}; strict=Val(true)
) where {N}
    (; solver, conditions, backends, representation) = implicit
    y, z = solver(x, args...)
    c = conditions(x, y, z, args...)
    if isnothing(backends)
        prep_A = nothing
        prep_Aᵀ = nothing
        prep_B = nothing
        prep_Bᵀ = nothing
    else
        if mode isa Union{ForwardMode,ForwardOrReverseMode}
            prep_A = prepare_A(
                representation, x, y, z, c, args...; conditions, backend=backends.y, strict
            )
            prep_B = prepare_B(
                representation, x, y, z, c, args...; conditions, backend=backends.x, strict
            )
        else
            prep_A = nothing
            prep_B = nothing
        end
        if mode isa Union{ReverseMode,ForwardOrReverseMode}
            prep_Aᵀ = prepare_Aᵀ(
                representation, x, y, z, c, args...; conditions, backend=backends.y, strict
            )
            prep_Bᵀ = prepare_Bᵀ(
                representation, x, y, z, c, args...; conditions, backend=backends.x, strict
            )
        else
            prep_Aᵀ = nothing
            prep_Bᵀ = nothing
        end
    end
    return ImplicitFunctionPreparation(eltype(x), prep_A, prep_Aᵀ, prep_B, prep_Bᵀ)
end

function prepare_A(
    ::MatrixRepresentation,
    x::AbstractArray,
    y::AbstractArray,
    z,
    c,
    args...;
    conditions,
    backend::AbstractADType,
    strict::Val,
)
    contexts = (Constant(x), Constant(z), map(Constant, args)...)
    return prepare_jacobian(Switch12(conditions), backend, y, contexts...; strict)
end

function prepare_A(
    ::OperatorRepresentation,
    x::AbstractArray,
    y::AbstractArray,
    z,
    c,
    args...;
    conditions,
    backend::AbstractADType,
    strict::Val,
)
    contexts = (Constant(x), Constant(z), map(Constant, args)...)
    return prepare_pushforward(
        Switch12(conditions), backend, y, (zero(y),), contexts...; strict
    )
end

function prepare_Aᵀ(
    ::MatrixRepresentation,
    x::AbstractArray,
    y::AbstractArray,
    z,
    c,
    args...;
    conditions,
    backend::AbstractADType,
    strict::Val,
)
    contexts = (Constant(x), Constant(z), map(Constant, args)...)
    return prepare_jacobian(Switch12(conditions), backend, y, contexts...; strict)
end

function prepare_Aᵀ(
    ::OperatorRepresentation,
    x::AbstractArray,
    y::AbstractArray,
    z,
    c,
    args...;
    conditions,
    backend::AbstractADType,
    strict::Val,
)
    contexts = (Constant(x), Constant(z), map(Constant, args)...)
    return prepare_pullback(
        Switch12(conditions), backend, y, (zero(c),), contexts...; strict
    )
end

function prepare_B(
    ::AbstractRepresentation,
    x::AbstractArray,
    y::AbstractArray,
    z,
    c,
    args...;
    conditions,
    backend::AbstractADType,
    strict::Val,
)
    contexts = (Constant(y), Constant(z), map(Constant, args)...)
    return prepare_pushforward(conditions, backend, x, (zero(x),), contexts...; strict)
end

function prepare_Bᵀ(
    ::AbstractRepresentation,
    x::AbstractArray,
    y::AbstractArray,
    z,
    c,
    args...;
    conditions,
    backend::AbstractADType,
    strict::Val,
)
    contexts = (Constant(y), Constant(z), map(Constant, args)...)
    return prepare_pullback(conditions, backend, x, (zero(c),), contexts...; strict)
end
