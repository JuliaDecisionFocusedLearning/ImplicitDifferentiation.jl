"""
    Forward{handle_byproduct,F}

Callable wrapper for a forward mapping `f::F`, which ensures that a byproduct `z(x)` is always returned in addition to `y(x)`.

The type parameter `handle_byproduct` is a boolean stating whether or not `f` natively returns `z(x)`.
"""
struct Forward{handle_byproduct,F}
    f::F
    function Forward{handle_byproduct}(f::F) where {handle_byproduct,F}
        return new{handle_byproduct,F}(f)
    end
end

function Base.show(io::IO, forward::Forward{handle_byproduct}) where {handle_byproduct}
    return print("Forward{$handle_byproduct}($(forward.f))")
end

"""
    forward(x; kwargs...)

Apply `forward.f` to `x`, returning a dummy byproduct `z(x)=0` if needed.
"""
(forward::Forward{true})(x; kwargs...) = forward.f(x; kwargs...)
(forward::Forward{false})(x; kwargs...) = (forward.f(x; kwargs...), 0)

"""
    Conditions{handle_byproduct,C}

Callable wrapper for conditions `c::C`, which ensures that a byproduct `z` is always accepted in addition to `x` and `y`.

The type parameter `handle_byproduct` is a boolean stating whether or not `c` natively accepts `z`.
"""
struct Conditions{handle_byproduct,C}
    c::C
    function Conditions{handle_byproduct}(c::C) where {handle_byproduct,C}
        return new{handle_byproduct,C}(c)
    end
end

function Base.show(
    io::IO, conditions::Conditions{handle_byproduct}
) where {handle_byproduct}
    return print("Conditions{$handle_byproduct}($(conditions.c))")
end

"""
    conditions(x, y, z; kwargs...)

Apply `conditions.c` to `(x, y, z)`, discarding `z` beforehand if needed.
"""
(conditions::Conditions{true})(x, y, z; kwargs...) = conditions.c(x, y, z; kwargs...)
(conditions::Conditions{false})(x, y, z; kwargs...) = conditions.c(x, y; kwargs...)

"""
    ImplicitFunction{handle_byproduct,FF<:Forward,CC<:Conditions,LS}

Differentiable wrapper for an implicit function defined by a forward mapping and a set of conditions.

# Constructors

    ImplicitFunction(f, c; linear_solver=gmres)
    ImplicitFunction(f, c, Val(handle_byproduct); linear_solver=gmres)

Construct an `ImplicitFunction` from a forward mapping `f` and conditions `c`, both of which are Julia callables.
While `f` does not not need to be compatible with automatic differentiation, `c` has to be.

# Details

- If `handle_byproduct=false` (the default), the forward mapping is `x -> y(x)` and the conditions are `c(x,y(x)) = 0`.
- If `handle_byproduct=true`, the forward mapping is `x -> (y(x),z(x))` and the conditions are `c(x,y(x),z(x)) = 0`. In this case, `z(x)` can contain additional information generated by the forward mapping, but beware that we consider it constant for differentiation purposes.

Given `x ∈ ℝⁿ` and `y ∈ ℝᵈ`, we need as many conditions as output dimensions: `c(x,y,z) ∈ ℝᵈ`. We can then compute the Jacobian of `y(⋅)` using the implicit function theorem:
```
∂₂c(x,y(x),z(x)) * ∂y(x) = -∂₁c(x,y(x),z(x))
```
This requires solving a linear system `A * J = -B`, where `A ∈ ℝᵈˣᵈ`, `B ∈ ℝᵈˣⁿ` and `J ∈ ℝᵈˣⁿ`.
The default linear solver is `Krylov.gmres`, but this can be changed with a keyword argument.

# Fields

- `forward::FF`: a wrapper of type [`Forward`](@ref) coherent with the value of `handle_byproduct`
- `conditions::FF`: a wrapper of type [`Conditions`](@ref) coherent with the value of `handle_byproduct`
- `linear_solver::LS`: a callable of the form `(A,b) -> (u,stats)` such that `Au = b` and `stats.solved ∈ {true,false}`, typically taken from [Krylov.jl](https://github.com/JuliaSmoothOptimizers/Krylov.jl)
"""
struct ImplicitFunction{
    handle_byproduct,FF<:Forward{handle_byproduct},CC<:Conditions{handle_byproduct},LS
}
    forward::FF
    conditions::CC
    linear_solver::LS

    function ImplicitFunction(
        f, c, ::Val{handle_byproduct}=Val(false); linear_solver=gmres
    ) where {handle_byproduct}
        forward = Forward{handle_byproduct}(f)
        conditions = Conditions{handle_byproduct}(c)
        return new{
            handle_byproduct,typeof(forward),typeof(conditions),typeof(linear_solver)
        }(
            forward, conditions, linear_solver
        )
    end
end

function Base.show(
    io::IO, implicit::ImplicitFunction{handle_byproduct}
) where {handle_byproduct}
    @unpack forward, conditions, linear_solver = implicit
    @unpack f = forward
    @unpack c = conditions
    return print(io, "ImplicitFunction{$handle_byproduct}($f, $c, $linear_solver)")
end

"""
    implicit(x::AbstractArray; kwargs...)
    implicit(x::AbstractArray, Val(return_byproduct), ; kwargs...)

Make an [`ImplicitFunction`](@ref) callable by applying the forward mapping `implicit.forward`.

- If `return_byproduct=false` (the default), this returns a single output `y(x)`.
- If `return_byproduct=true`, this returns a couple of outputs `(y(x),z(x))`.

The argument `return_byproduct` is independent from the type parameter `handle_byproduct` in `ImplicitFunction`, so any combination is possible.
"""
function (implicit::ImplicitFunction)(
    x::AbstractArray, ::Val{return_byproduct}=Val(false); kwargs...
) where {return_byproduct}
    y, z = implicit.forward(x, ; kwargs...)
    if return_byproduct
        return (y, z)
    else
        return y
    end
end
