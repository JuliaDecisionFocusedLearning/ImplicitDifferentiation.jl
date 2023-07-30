"""
    Forward{byproduct,F}

Callable wrapper for a forward mapping `f::F`, which ensures that a byproduct `z(x)` is always returned in addition to `y(x)`.

The type parameter `byproduct` is a boolean stating whether or not `f` natively returns `z(x)`.
"""
struct Forward{byproduct,F}
    f::F
    function Forward{byproduct}(f::F) where {byproduct,F}
        return new{byproduct,F}(f)
    end
end

function Base.show(io::IO, forward::Forward{byproduct}) where {byproduct}
    return print(io, "Forward{$byproduct}($(forward.f))")
end

function (forward::Forward{true})(x; kwargs...)
    yz = forward.f(x; kwargs...)
    if !(yz isa Tuple && length(yz) == 2)
        throw(
            ArgumentError(
                "The forward function does not handle the by-product correctly. The forward function should return a tuple of 2 outputs, the main output and the byproduct.",
            ),
        )
    end
    return yz
end

function (forward::Forward{false})(x; kwargs...)
    y = forward.f(x; kwargs...)
    z = 0
    return y, z
end

handles_byproduct(::Forward{byproduct}) where {byproduct} = byproduct
