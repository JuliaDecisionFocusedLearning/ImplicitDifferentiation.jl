function (implicit::ImplicitFunction)(x::AbstractArray, args::Vararg{Any,N}) where {N}
    return implicit(ImplicitFunctionPreparation(), x, args...)
end

function (implicit::ImplicitFunction)(
    ::ImplicitFunctionPreparation, x::AbstractArray, args::Vararg{Any,N}
) where {N}
    return implicit.solver(x, args...)
end
