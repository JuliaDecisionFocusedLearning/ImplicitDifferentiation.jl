function simplex_projection_and_support(z::AbstractVector{<:Real})
    d = length(z)
    z_sorted = sort(z; rev=true)
    z_sorted_cumsum = cumsum(z_sorted)
    k = maximum(j for j in 1:d if (1 + j * z_sorted[j]) > z_sorted_cumsum[j])
    τ = (z_sorted_cumsum[k] - 1) / k
    p = max.(z .- τ, 0)
    s = [Int(p[i] > eps()) for i in 1:d]
    return p, s
end;

function simplex_projection(z::AbstractVector{<:Real})
    p, _ = simplex_projection_and_support(z)
    return p
end;

function ChainRulesCore.rrule(::typeof(simplex_projection), z::AbstractVector{<:Real})
    p, s = simplex_projection_and_support(z)
    S = sum(s)
    function simplex_projection_pullback(dp)
        vjp = s .* (dp .- (dp's) / S)
        return (NoTangent(), vjp)
    end
    return p, simplex_projection_pullback
end;
