
function flatten_similar(x::Real, y::Real)
    v = [x]
    unflatten_to_Real(v) = only(v)
    return v, unflatten_to_Real
end

flatten_similar(x::Vector{<:Real}, y::Vector{<:Real}) = (identity.(x), identity)

function flatten_similar(x::Tuple, y::Tuple)
    @info "Flatten similar tup-tup" x y
    x_vecs_and_unflattens = map((xᵢ, yᵢ) -> flatten_similar(xᵢ, yᵢ), zip(x, y))
    x_vecs, unflattens = first.(x_vecs_and_unflattens), last.(x_vecs_and_unflattens)
    lengths = map(_length, x_vecs)
    sizes = _cumsum(lengths)
    function unflatten_to_Tuple(v)
        map(unflattens, lengths, sizes) do un, l, s
            return un(v[(s - l + 1):s])
        end
    end
    return reduce(vcat, x_vecs), unflatten_to_Tuple
end

function flatten_similar(x::AbstractVector, y::AbstractVector)
    @info "Flatten similar vec-vec" x y
    x_vecs_and_unflattens = map(
        (xᵢ, yᵢ) -> flatten_similar(xᵢ, yᵢ), zip(identity.(x), identity.(y))
    )
    x_vecs, unflattens = first.(x_vecs_and_unflattens), last.(x_vecs_and_unflattens)
    sizes = _cumsum(map(_length, x_vecs))
    function unflatten_to_AbstractVector(v)
        return map(unflattens, lengths, sizes) do un, l, s
            return un(v[(s - l + 1):s])
        end
    end
    return reduce(vcat, x_vecs), unflatten_to_AbstractVector
end

function flatten_similar(x::AbstractArray, y::AbstractArray)
    @info "Flatten similar arr-arr" x y
    x_vec, unflatten = flatten_similar(vec(identity.(x)), vec(identity.(y)))
    unflatten_to_AbstractArray(v) = reshape(unflatten(v), size(x))
    return identity.(x_vec), unflatten_to_AbstractArray
end

# Zygote can return a sparse vector co-tangent
# even if the input is a vector. This is causing
# issues in the rrule definition of Unflatten
flatten_similar(x::SparseVector, y::SparseVector) = flatten_similar(Array(x), Array(y))

function flatten_similar(x::SparseMatrixCSC, y::SparseMatrixCSC)
    x_vec, unflatten = flatten_similar(x.nzval, y.nzval)
    function unflatten_to_SparseMatrixSCS(v)
        return SparseMatrixCSC(x.m, x.n, x.colptr, x.rowval, unflatten(v))
    end
    return identity.(x_vec), unflatten_to_SparseMatrixSCS
end

function flatten_similar(x::Tangent, y)
    @info "Flatten similar tan-any" x y
    return flatten_similar(ntfromstruct(x).backing, y)
end

function flatten_similar(x::NamedTuple, y)
    @info "Flatten similar nt-any" x y
    return flatten_similar(x, ntfromstruct(y))
end

function flatten_similar(x::NamedTuple, y::NamedTuple)
    @info "Flatten similar nt-nt" x y
    x_vec, unflatten = flatten_similar(values(x), values(y))
    function unflatten_to_NamedTuple(v)
        x_values = unflatten(v)
        return NamedTuple{keys(x)}(x_values)
    end
    return identity.(x_vec), unflatten_to_NamedTuple
end

function flatten_similar(x::AbstractDict, y::AbstractDict, ks=collect(keys(x)))
    @info "Flatten similar dict-dict" x y
    x_ordered = OrderedDict(k => x[k] for k in ks)
    y_ordered = OrderedDict(k => y[k] for k in ks)
    x_vec, unflatten = flatten_similar(
        identity.(collect(values(x_ordered))), identity.(collect(values(y_ordered)))
    )
    function unflatten_to_Dict(v)
        x_values = unflatten(v)
        return OrderedDict(key => x_values[n] for (n, key) in enumerate(ks))
    end
    return identity.(x_vec), unflatten_to_Dict
end

function flatten_similar(x, y)
    @info "Flatten similar any-any" x y
    v, un = flatten_similar(ntfromstruct(x), ntfromstruct(y))
    return identity.(v), Unflatten(y, z -> structfromnt(typeof(x), un(z)))
end

function flatten_similar(::Nothing, y)
    y_vec, unflatten = flatten(y)
    return zero(y_vec), unflatten
end

function flatten_similar(::NoTangent, y)
    y_vec, unflatten = flatten(y)
    return zero(y_vec), unflatten
end

function flatten_similar(::ZeroTangent, y)
    y_vec, unflatten = flatten(y)
    return zero(y_vec), unflatten
end

function flatten_similar(::Tuple{}, ::Any)
    return Float64[], _ -> ()
end
