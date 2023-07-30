"""
    Conditions{byproduct,C}

Callable wrapper for conditions `c::C`, which ensures that a byproduct `z` is always accepted in addition to `x` and `y`.

The type parameter `byproduct` is a boolean stating whether or not `c` natively accepts `z`.
"""
struct Conditions{byproduct,C}
    c::C
    function Conditions{byproduct}(c::C) where {byproduct,C}
        return new{byproduct,C}(c)
    end
end

function Base.show(io::IO, conditions::Conditions{byproduct}) where {byproduct}
    return print(io, "Conditions{$byproduct}($(conditions.c))")
end

(conditions::Conditions{true})(x, y, z; kwargs...) = conditions.c(x, y, z; kwargs...)
(conditions::Conditions{false})(x, y, z; kwargs...) = conditions.c(x, y; kwargs...)

handles_byproduct(::Conditions{byproduct}) where {byproduct} = byproduct
