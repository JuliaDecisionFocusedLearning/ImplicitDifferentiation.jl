
function flatten_similar(::Real, x::Real)
    v = [x]
    unflatten_to_Real(v) = only(v)
    return v, unflatten_to_Real
end

flatten_similar(::Vector{<:Real}, x::Vector{<:Real}) = (identity.(x), identity)

function flatten_similar(x1::AbstractVector, x2::AbstractVector)
    x_vecs_and_backs = map((val) -> flatten_similar(val[1], val[2]), zip(identity.(x1), identity.(x2)))
    x_vecs, backs = first.(x_vecs_and_backs), last.(x_vecs_and_backs)
    function Vector_from_vec(x_vec)
        sz = _cumsum(map(_length, x_vecs))
        x_Vec = [backs[n](x_vec[sz[n] - _length(x_vecs[n]) + 1:sz[n]]) for n in eachindex(x2)]
        return x_Vec
    end
    return reduce(vcat, x_vecs), Vector_from_vec
end

function flatten_similar(x1::AbstractArray, x2::AbstractArray)
    x_vec, from_vec = flatten_similar(vec(identity.(x1)), vec(identity.(x2)))
    Array_from_vec(x_vec) = reshape(from_vec(x_vec), size(x2))
    return identity.(x_vec), Array_from_vec
end

# Zygote can return a sparse vector co-tangent
# even if the input is a vector. This is causing
# issues in the rrule definition of Unflatten
flatten_similar(x1::SparseVector, x2::SparseVector) = flatten_similar(Array(x1), Array(x2))

# function flatten_similar(x1::JuMP.Containers.DenseAxisArray, x2::NamedTuple)
#     x_vec, from_vec = flatten_similar(vec(identity.(x1.data)), vec(identity.(x2.data)))
#     Array_from_vec(x_vec) = JuMP.Containers.DenseAxisArray(reshape(from_vec(x_vec), size(x2)), axes(x2)...)
#     return identity.(x_vec), Array_from_vec
# end

# function flatten_similar(x1::JuMP.Containers.DenseAxisArray, x2::JuMP.Containers.DenseAxisArray)
#     x_vec, from_vec = flatten_similar(vec(identity.(x1.data)), vec(identity.(x2.data)))
#     Array_from_vec(x_vec) = JuMP.Containers.DenseAxisArray(reshape(from_vec(x_vec), size(x2)), axes(x2)...)
#     return identity.(x_vec), Array_from_vec
# end

function flatten_similar(x1::SparseMatrixCSC, x2::SparseMatrixCSC)
    x_vec, from_vec = flatten_similar(x1.nzval, x2.nzval)
    Array_from_vec(x_vec) = SparseMatrixCSC(x.m, x.n, x.colptr, x.rowval, from_vec(x_vec))
    return identity.(x_vec), Array_from_vec
end

function flatten_similar(x1::Tuple, x2::Tuple)
    x_vecs_and_backs = map(val -> flatten_similar(val[1], val[2]), zip(x1, x2))
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

function flatten_similar(x1, x2::Tangent)
    flatten_similar(x1, ntfromstruct(x2).backing)
end
function flatten_similar(x1, x2::NamedTuple)
    flatten_similar(ntfromstruct(x1), x2)
end

function flatten_similar(x1::NamedTuple, x2::NamedTuple)
    x_vec, unflatten = flatten_similar(values(x1), values(x2))
    function unflatten_to_NamedTuple(v)
        v_vec_vec = unflatten(v)
        return NamedTuple{keys(x1)}(v_vec_vec)
    end
    return identity.(x_vec), unflatten_to_NamedTuple
end

function flatten_similar(d1::AbstractDict, d2::AbstractDict, ks = collect(keys(d2)))
    _d1 = OrderedDict(k => d1[k] for k in ks)
    _d2 = OrderedDict(k => d2[k] for k in ks)
    d_vec, unflatten = flatten_similar(identity.(collect(values(_d1))), identity.(collect(values(_d2))))
    function unflatten_to_Dict(v)
        v_vec_vec = unflatten(v)
        return OrderedDict(key => v_vec_vec[n] for (n, key) in enumerate(ks))
    end
    return identity.(d_vec), unflatten_to_Dict
end

function flatten_similar(x1, x2)
    v, un = flatten_similar(ntfromstruct(x1), ntfromstruct(x2))
    return identity.(v), Unflatten(x1, y -> structfromnt(typeof(x2), un(y)))
end


function flatten_similar(x, ::Nothing)
    t = flatten(x)
    return zero(t[1]), Base.tail(t)
end
function flatten_similar(x, ::NoTangent)
    t = flatten(x)
    return zero(t[1]), Base.tail(t)
end
function flatten_similar(x, ::ZeroTangent)
    t = flatten(x)
    return zero(t[1]), Base.tail(t)
end
function flatten_similar(::Any, ::Tuple{})
    return Float64[], _ -> ()
end
