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


function flatten(x)
    v, un = flatten(ntfromstruct(x))
    return identity.(v), Unflatten(x, y -> structfromnt(typeof(x), un(y)))
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
