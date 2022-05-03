## Flatten

function flatten_similar(x::R, y::T) where {T<:Real,R<:Real}
    x_vec = T[x]
    unflatten_to_Real(x_vec) = convert(R, only(x_vec))
    return x_vec, unflatten_to_Real
end

function flatten_similar(x::Vector{R}, y::Vector{T}) where {T<:Real,R<:Real}
    return (Vector{T}(x), Vector{R})
end

function flatten_similar(x::AbstractVector, y::AbstractVector)
    x_vecs_and_unflattens = map((xᵢ, yᵢ) -> flatten_similar(xᵢ, yᵢ), zip(x, y))
    x_vecs, unflattens = first.(x_vecs_and_unflattens), last.(x_vecs_and_unflattens)
    x_vec = reduce(vcat, x_vecs)
    lengths = map(_length, x_vecs)
    sizes = _cumsum(lengths)
    function unflatten_to_AbstractVector(x_vec)
        x_AbstractVector = map(unflattens, lengths, sizes) do unflatten, l, s
            return unflatten(x_vec[(s - l + 1):s])
        end
        return oftype(x, x_AbstractVector)
    end
    return x_vec, unflatten_to_AbstractVector
end

function flatten_similar(x::AbstractArray, y::AbstractArray)
    x_vec, unflatten = flatten_similar(vec(x), vec(y))
    function unflatten_to_AbstractArray(x_vec)
        x_AbstractArray = reshape(unflatten(x_vec), size(y))
        return oftype(x, x_AbstractArray)
    end
    return x_vec, unflatten_to_AbstractArray
end

#=
Zygote can return a sparse vector co-tangent even if the input is a vector. This is causing issues in the rrule definition of Unflatten
=#
flatten_similar(x::SparseVector, y::SparseVector) = flatten_similar(Array(x), Array(y))

function flatten_similar(x::SparseMatrixCSC, y::SparseMatrixCSC)
    x_vec, unflatten = flatten_similar(x.nzval, y.nzval)
    function unflatten_to_SparseMatrixCSC(x_vec)
        return SparseMatrixCSC(x.m, x.n, x.colptr, x.rowval, unflatten(x_vec))
    end
    return x_vec, unflatten_to_SparseMatrixCSC
end

function flatten_similar(x::Tuple, y::Tuple)
    x_vecs_and_unflattens = map((xᵢ, yᵢ) -> flatten_similar(xᵢ, yᵢ), zip(x, y))
    x_vecs, unflattens = first.(x_vecs_and_unflattens), last.(x_vecs_and_unflattens)
    x_vec = reduce(vcat, x_vecs)
    lengths = map(_length, x_vecs)
    sizes = _cumsum(lengths)
    function unflatten_to_Tuple(x_vec)
        x_Tuple = map(unflattens, lengths, sizes) do unflatten, l, s
            return unflatten(x_vec[(s - l + 1):s])
        end
        return oftype(x, x_Tuple)
    end
    return x_vec, unflatten_to_Tuple
end

function flatten_similar(x::NamedTuple, y)
    return flatten_similar(x, ntfromstruct(y))
end

function flatten_similar(x::Tangent, y)
    return flatten_similar(ntfromstruct(x).backing, y)
end

function flatten_similar(x::NamedTuple, y::NamedTuple)
    x_vec, unflatten = flatten_similar(values(x), values(y))
    function unflatten_to_NamedTuple(x_vec)
        x_vec_vals = unflatten(x_vec)
        return NamedTuple{keys(x)}(x_vec_vals)
    end
    return x_vec, unflatten_to_NamedTuple
end
