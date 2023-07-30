"""
    Conditions{byproduct,C,B}

Callable wrapper for conditions `c`, which ensures that a byproduct `z` is always accepted in addition to `x` and `y`.

The type parameter `byproduct` is a boolean stating whether or not `c` natively accepts `z`.

# Fields

- `c::C`: Callable returning an array that must be equal to zero.
- `backend::B`: Autodiff backend compatible with AbstractDifferentiation.jl, which will be used to differentiate the conditions. It defaults to `nothing`, which means the conditions will use the same backend as the implicit function they belong to.
"""
struct Conditions{byproduct,C,B<:Union{Nothing,<:AbstractBackend}}
    c::C
    backend::B
    function Conditions{byproduct}(c::C, backend::B=nothing) where {byproduct,C,B}
        return new{byproduct,C,B}(c, backend)
    end
end

function Base.show(io::IO, conditions::Conditions{byproduct}) where {byproduct}
    return print(io, "Conditions{$byproduct}($(conditions.c), $(conditions.backend))")
end

(conditions::Conditions{true})(x, y, z; kwargs...) = conditions.c(x, y, z; kwargs...)
(conditions::Conditions{false})(x, y, z; kwargs...) = conditions.c(x, y; kwargs...)

handles_byproduct(::Conditions{byproduct}) where {byproduct} = byproduct
