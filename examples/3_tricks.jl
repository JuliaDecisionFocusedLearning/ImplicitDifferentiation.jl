# # Tricks

#=
We demonstrate several features that may come in handy for some users.
=#

using ComponentArrays
using ForwardDiff
using ImplicitDifferentiation
using LinearAlgebra
using Test  #src
using Zygote

# ## ComponentArrays

# For when you need derivatives with respect to multiple inputs or outputs.

function forward_components_aux(a::AbstractVector, b::AbstractVector, m::Number)
    d = m * sqrt.(a)
    e = sqrt.(b)
    return d, e
end

function conditions_components_aux(a, b, m, d, e)
    c_d = (d ./ m) .^ 2 .- a
    c_e = (e .^ 2) .- b
    return c_d, c_e
end;

# You can use `ComponentVector` from [ComponentArrays.jl](https://github.com/jonniedie/ComponentArrays.jl) as an intermediate storage.

function forward_components(x::ComponentVector)
    d, e = forward_components_aux(x.a, x.b, x.m)
    y = ComponentVector(; d=d, e=e)
    z = nothing
    return y, z
end

function conditions_components(x::ComponentVector, y::ComponentVector, _z)
    c_d, c_e = conditions_components_aux(x.a, x.b, x.m, y.d, y.e)
    c = ComponentVector(; c_d=c_d, c_e=c_e)
    return c
end;

# And build your implicit function like so:

implicit_components = ImplicitFunction(
    forward_components, conditions_components; strict=Val(false)
);

# Now we're good to go.

a, b, m = [1.0, 2.0], [3.0, 4.0, 5.0], 6.0
x = ComponentVector(; a=a, b=b, m=m)
implicit_components(x)

# And it works with both ForwardDiff.jl and Zygote.jl

ForwardDiff.jacobian(first ∘ implicit_components, x)
J = ForwardDiff.jacobian(first ∘ forward_components, x)  #src
@test ForwardDiff.jacobian(first ∘ implicit_components, x) ≈ J  #src

#-

Zygote.jacobian(first ∘ implicit_components, x)[1]
@test Zygote.jacobian(first ∘ implicit_components, x)[1] ≈ J  #src

#- The full differentiable pipeline looks like this

function full_pipeline(a, b, m)
    x = ComponentVector(; a=a, b=b, m=m)
    y, _ = implicit_components(x)
    return y.d, y.e
end;
