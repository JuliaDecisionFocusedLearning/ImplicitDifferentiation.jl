# WARNING: type piracy ongoing

function ParameterHandling.flatten(::Type{T}, tang::Tangent{P, B}) where {T, P, B}
    tang_tup = ntfromstruct(tang).backing
    v, unflatten = flatten(T, tang_tup)
    unflatten_to_Tangent(v) = Tangent{P, B}(unflatten(v))
    return v, unflatten_to_Tangent
end
