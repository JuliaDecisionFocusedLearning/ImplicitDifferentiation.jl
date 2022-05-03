_length(x) = length(x)
_length(::Nothing) = 0

_cumsum(x) = cumsum(x)
if VERSION < v"1.5"
    _cumsum(x::Tuple) = (_cumsum(collect(x))...,)
end

_zero(x::Real) = zero(x)
_zero(x::AbstractArray) = _zero.(x)
_zero(x::AbstractDict) = Dict(keys(x) .=> map(_zero, values(x)))
_zero(x::NamedTuple) = map(_zero, x)
_zero(x::Tuple) = map(_zero, x)
_zero(x) = structfromnt(typeof(x), _zero(ntfromstruct(x)))

function _merge(d1, d2::AbstractDict)
    _d = OrderedDict(k => _zero(v) for (k, v) in d1)
    return sort!(merge(_d, OrderedDict(d2)))
end

_merge(::Any, d2) = d2

function _build_ordered_dict(vals, keys)
    OrderedDict(key => vals[n] for (n, key) in enumerate(keys))
end

function ChainRulesCore.rrule(::typeof(_build_ordered_dict), vals, keys)
    _build_ordered_dict(vals, keys), Δ -> begin
        NoTangent(), values(Δ), NoTangent()
    end
end
