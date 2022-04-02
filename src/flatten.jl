# Warning: type piracy

function ParameterHandling.flatten(::Type{T}, tang::Tangent{P,B}) where {T,P,B}
    tang_tup = ntfromstruct(tang).backing
    v, unflatten = flatten(T, tang_tup)
    unflatten_to_Tangent(v) = Tangent{P,B}(unflatten(v))
    return v, unflatten_to_Tangent
end

function ParameterHandling.flatten(::Type{T}, tang::NoTangent) where {T}
    v = T[]
    unflatten_to_NoTangent(v) = NoTangent()
    return v, unflatten_to_NoTangent
end

function ParameterHandling.flatten(::Type{T}, tang::ZeroTangent) where {T}
    v = T[]
    unflatten_to_ZeroTangent(v) = ZeroTangent()
    return v, unflatten_to_ZeroTangent
end
