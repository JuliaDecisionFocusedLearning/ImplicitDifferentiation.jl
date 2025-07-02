function (implicit::ImplicitFunction)(x::AbstractArray, args::Vararg{Any,N}) where {N}
    return implicit(ImplicitFunctionPreparation(eltype(x)), x, args...)
end

function (implicit::ImplicitFunction)(
    ::ImplicitFunctionPreparation{R}, x::AbstractArray{R}, args::Vararg{Any,N}
) where {R<:Real,N}
    return implicit.solver(x, args...)
end
