"""
    simplex_projection_and_support(z)

Compute the Euclidean projection onto the probability simplex and the set of indices where it is nonzero.

See <https://arxiv.org/abs/1602.02068> for details.
"""
function simplex_projection_and_support(z::AbstractVector{R}) where {R<:Real}
    d = length(z)
    z_sorted = sort(z; rev=true)
    z_sorted_cumsum = cumsum(z_sorted)
    k = maximum(j for j in 1:d if (1 + j * z_sorted[j]) > z_sorted_cumsum[j])
    τ = (z_sorted_cumsum[k] - 1) / k
    p = Vector{R}(undef, d)
    s = Vector{Int}(undef, d)
    for i in 1:d
        p[i] = max(z[i] - τ, zero(R))
        s[i] = Int(!iszero(p[i]))
    end
    return p, s
end;

"""
    simplex_projection(z)

Compute the Euclidean projection onto the probability simplex.
"""
function simplex_projection(z::AbstractVector{<:Real})
    p, _ = simplex_projection_and_support(z)
    return p
end;

"""
    rrule(::typeof(simplex_projection), z)

Custom reverse rule for [`simplex_projection`](@ref) which bypasses the sorting step.

See <https://arxiv.org/abs/1602.02068> for details.
"""
function ChainRulesCore.rrule(::typeof(simplex_projection), z::AbstractVector{<:Real})
    p, s = simplex_projection_and_support(z)
    S = sum(s)
    function simplex_projection_pullback(dp)
        vjp = s .* (dp .- (dp's) / S)
        return (NoTangent(), vjp)
    end
    return p, simplex_projection_pullback
end;
