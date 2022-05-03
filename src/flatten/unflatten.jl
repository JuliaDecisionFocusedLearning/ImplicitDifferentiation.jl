struct Unflatten{X,F} <: Function
    x::X
    unflatten::F
end

(un::Unflatten)(x) = un.unflatten(x)

function ChainRulesCore.rrule(un::Unflatten, v)
    x = un(v)
    return x, Δ -> begin
        _Δ = _merge(x, Δ)
        return (NoTangent(), flatten_similar(un.x, _Δ)[1])
    end
end
