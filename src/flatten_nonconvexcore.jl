#=
Adapted from NonconvexCore.jl with the following license.

MIT License

Copyright (c) 2021 Mohamed Tarek <mohamed82008@gmail.com> and contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
=#

#=
The code in NonconvexCore.jl was itself adapted from ParameterHandling.jl with the following license.

Copyright (c) 2020 Invenia Technical Computing Corporation

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
=#

"""
    flatten(x)

Returns a "flattened" representation of `x` as a vector of real numbers, and a function
`unflatten` that takes a vector of reals of the same length and returns an object of the
same type as `x`.

`unflatten` is the inverse of `flatten`, so
```julia
julia> x = (randn(5), 5.0, (a=5.0, b=randn(2, 3)));

julia> v, unflatten = flatten(x);

julia> x == unflatten(v)
true
```
"""
function flatten end

maybeflatten(x::Real) = x
maybeflatten(x) = flatten(x)

function flatten(x::Real)
    v = [x]
    unflatten_to_Real(v) = only(v)
    return v, unflatten_to_Real
end

flatten(x::Vector{<:Real}) = (identity.(x), identity)

function flatten(x::AbstractVector)
    x_vecs_and_backs = map(val -> flatten(val), identity.(x))
    x_vecs, backs = first.(x_vecs_and_backs), last.(x_vecs_and_backs)
    function Vector_from_vec(x_vec)
        sz = _cumsum(map(_length, x_vecs))
        x_Vec = [backs[n](x_vec[sz[n] - _length(x_vecs[n]) + 1:sz[n]]) for n in eachindex(x)]
        return x_Vec
    end
    return reduce(vcat, x_vecs), Vector_from_vec
end

function flatten(x::AbstractArray)
    x_vec, from_vec = flatten(vec(identity.(x)))
    Array_from_vec(x_vec) = reshape(from_vec(x_vec), size(x))
    return identity.(x_vec), Array_from_vec
end

# Zygote can return a sparse vector co-tangent
# even if the input is a vector. This is causing
# issues in the rrule definition of Unflatten
flatten(x::SparseVector) = flatten(Array(x))

# function flatten(x::JuMP.Containers.DenseAxisArray)
#     x_vec, from_vec = flatten(vec(identity.(x.data)))
#     Array_from_vec(x_vec) = JuMP.Containers.DenseAxisArray(reshape(from_vec(x_vec), size(x)), axes(x)...)
#     return identity.(x_vec), Array_from_vec
# end

function flatten(x::SparseMatrixCSC)
    x_vec, from_vec = flatten(x.nzval)
    Array_from_vec(x_vec) = SparseMatrixCSC(x.m, x.n, x.colptr, x.rowval, from_vec(x_vec))
    return identity.(x_vec), Array_from_vec
end

function flatten(x::Tuple)
    x_vecs_and_backs = map(val -> flatten(val), x)
    x_vecs, x_backs = first.(x_vecs_and_backs), last.(x_vecs_and_backs)
    lengths = map(_length, x_vecs)
    sz = _cumsum(lengths)
    function unflatten_to_Tuple(v)
        map(x_backs, lengths, sz) do x_back, l, s
            return x_back(v[s - l + 1:s])
        end
    end
    return reduce(vcat, x_vecs), unflatten_to_Tuple
end

function flatten(x::NamedTuple)
    x_vec, unflatten = flatten(values(x))
    function unflatten_to_NamedTuple(v)
        v_vec_vec = unflatten(v)
        return NamedTuple{keys(x)}(v_vec_vec)
    end
    return identity.(x_vec), unflatten_to_NamedTuple
end

function flatten(d::AbstractDict, ks = collect(keys(d)))
    _d = OrderedDict(k => d[k] for k in ks)
    d_vec, unflatten = flatten(identity.(collect(values(_d))))
    function unflatten_to_Dict(v)
        v_vec_vec = unflatten(v)
        return _build_ordered_dict(v_vec_vec, keys(_d))
    end
    return identity.(d_vec), unflatten_to_Dict
end
function _build_ordered_dict(vals, keys)
    OrderedDict(key => vals[n] for (n, key) in enumerate(keys))
end
function ChainRulesCore.rrule(::typeof(_build_ordered_dict), vals, keys)
    _build_ordered_dict(vals, keys), Δ -> begin
        NoTangent(), values(Δ), NoTangent()
    end
end

function flatten(x)
    v, un = flatten(ntfromstruct(x))
    return identity.(v), Unflatten(x, y -> structfromnt(typeof(x), un(y)))
end

function zygote_flatten(::Real, x::Real)
    v = [x]
    unflatten_to_Real(v) = only(v)
    return v, unflatten_to_Real
end

zygote_flatten(::Vector{<:Real}, x::Vector{<:Real}) = (identity.(x), identity)

function zygote_flatten(x1::AbstractVector, x2::AbstractVector)
    x_vecs_and_backs = map((val) -> zygote_flatten(val[1], val[2]), zip(identity.(x1), identity.(x2)))
    x_vecs, backs = first.(x_vecs_and_backs), last.(x_vecs_and_backs)
    function Vector_from_vec(x_vec)
        sz = _cumsum(map(_length, x_vecs))
        x_Vec = [backs[n](x_vec[sz[n] - _length(x_vecs[n]) + 1:sz[n]]) for n in eachindex(x2)]
        return x_Vec
    end
    return reduce(vcat, x_vecs), Vector_from_vec
end

function zygote_flatten(x1::AbstractArray, x2::AbstractArray)
    x_vec, from_vec = zygote_flatten(vec(identity.(x1)), vec(identity.(x2)))
    Array_from_vec(x_vec) = reshape(from_vec(x_vec), size(x2))
    return identity.(x_vec), Array_from_vec
end

# Zygote can return a sparse vector co-tangent
# even if the input is a vector. This is causing
# issues in the rrule definition of Unflatten
zygote_flatten(x1::SparseVector, x2::SparseVector) = zygote_flatten(Array(x1), Array(x2))

# function zygote_flatten(x1::JuMP.Containers.DenseAxisArray, x2::NamedTuple)
#     x_vec, from_vec = zygote_flatten(vec(identity.(x1.data)), vec(identity.(x2.data)))
#     Array_from_vec(x_vec) = JuMP.Containers.DenseAxisArray(reshape(from_vec(x_vec), size(x2)), axes(x2)...)
#     return identity.(x_vec), Array_from_vec
# end

# function zygote_flatten(x1::JuMP.Containers.DenseAxisArray, x2::JuMP.Containers.DenseAxisArray)
#     x_vec, from_vec = zygote_flatten(vec(identity.(x1.data)), vec(identity.(x2.data)))
#     Array_from_vec(x_vec) = JuMP.Containers.DenseAxisArray(reshape(from_vec(x_vec), size(x2)), axes(x2)...)
#     return identity.(x_vec), Array_from_vec
# end

function zygote_flatten(x1::SparseMatrixCSC, x2::SparseMatrixCSC)
    x_vec, from_vec = zygote_flatten(x1.nzval, x2.nzval)
    Array_from_vec(x_vec) = SparseMatrixCSC(x.m, x.n, x.colptr, x.rowval, from_vec(x_vec))
    return identity.(x_vec), Array_from_vec
end

function zygote_flatten(x1::Tuple, x2::Tuple)
    x_vecs_and_backs = map(val -> zygote_flatten(val[1], val[2]), zip(x1, x2))
    x_vecs, x_backs = first.(x_vecs_and_backs), last.(x_vecs_and_backs)
    lengths = map(_length, x_vecs)
    sz = _cumsum(lengths)
    function unflatten_to_Tuple(v)
        map(x_backs, lengths, sz) do x_back, l, s
            return x_back(v[s - l + 1:s])
        end
    end
    return reduce(vcat, x_vecs), unflatten_to_Tuple
end

function zygote_flatten(x1, x2::Tangent)
    zygote_flatten(x1, ntfromstruct(x2).backing)
end
function zygote_flatten(x1, x2::NamedTuple)
    zygote_flatten(ntfromstruct(x1), x2)
end

function zygote_flatten(x1::NamedTuple, x2::NamedTuple)
    x_vec, unflatten = zygote_flatten(values(x1), values(x2))
    function unflatten_to_NamedTuple(v)
        v_vec_vec = unflatten(v)
        return NamedTuple{keys(x1)}(v_vec_vec)
    end
    return identity.(x_vec), unflatten_to_NamedTuple
end

function zygote_flatten(d1::AbstractDict, d2::AbstractDict, ks = collect(keys(d2)))
    _d1 = OrderedDict(k => d1[k] for k in ks)
    _d2 = OrderedDict(k => d2[k] for k in ks)
    d_vec, unflatten = zygote_flatten(identity.(collect(values(_d1))), identity.(collect(values(_d2))))
    function unflatten_to_Dict(v)
        v_vec_vec = unflatten(v)
        return OrderedDict(key => v_vec_vec[n] for (n, key) in enumerate(ks))
    end
    return identity.(d_vec), unflatten_to_Dict
end

function zygote_flatten(x1, x2)
    v, un = zygote_flatten(ntfromstruct(x1), ntfromstruct(x2))
    return identity.(v), Unflatten(x1, y -> structfromnt(typeof(x2), un(y)))
end

_length(x) = length(x)
_length(::Nothing) = 0

function ChainRulesCore.rrule(::typeof(flatten), x)
    d_vec, un = flatten(x)
    return (d_vec, un), Δ -> begin
        (NoTangent(), un(Δ[1]), NoTangent())
    end
end
function ChainRulesCore.rrule(::typeof(flatten), d::AbstractDict, ks)
    _d = OrderedDict(k => d[k] for k in ks)
    d_vec, un = flatten(_d, ks)
    return (d_vec, un), Δ -> begin
        (NoTangent(), un(Δ[1]), NoTangent())
    end
end

struct Unflatten{X, F} <: Function
    x::X
    unflatten::F
end
(f::Unflatten)(x) = f.unflatten(x)

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

function ChainRulesCore.rrule(un::Unflatten, v)
    x = un(v)
    return x, Δ -> begin
        _Δ = _merge(x, Δ)
        return (NoTangent(), zygote_flatten(un.x, _Δ)[1])
    end
end

function flatten(::Nothing)
    return Float64[], _ -> nothing
end
function flatten(::NoTangent)
    return Float64[], _ -> NoTangent()
end
function flatten(::ZeroTangent)
    return Float64[], _ -> ZeroTangent()
end
function flatten(::Tuple{})
    return Float64[], _ -> ()
end

function zygote_flatten(x, ::Nothing)
    t = flatten(x)
    return zero(t[1]), Base.tail(t)
end
function zygote_flatten(x, ::NoTangent)
    t = flatten(x)
    return zero(t[1]), Base.tail(t)
end
function zygote_flatten(x, ::ZeroTangent)
    t = flatten(x)
    return zero(t[1]), Base.tail(t)
end
function zygote_flatten(::Any, ::Tuple{})
    return Float64[], _ -> ()
end

macro constructor(T)
    return flatten_expr(T, T)
end
macro constructor(T, C)
    return flatten_expr(T, C)
end
flatten_expr(T, C) = quote
    function NonconvexCore.flatten(x::$(esc(T)))
        v, un = flatten(ntfromstruct(x))
        return identity.(v), Unflatten(x, y -> structfromnt($(esc(C)), un(y)))
    end
    function NonconvexCore.zygote_flatten(x1::$(esc(T)), x2::$(esc(T)))
        v, un = zygote_flatten(ntfromstruct(x1), ntfromstruct(x2))
        return identity.(v), Unflatten(x2, y -> structfromnt($(esc(C)), un(y)))
    end
    NonconvexCore._zero(x::$(esc(T))) = structfromnt($(esc(C)), _zero(ntfromstruct(x)))
end

_cumsum(x) = cumsum(x)
if VERSION < v"1.5"
    _cumsum(x::Tuple) = (_cumsum(collect(x))..., )
end
