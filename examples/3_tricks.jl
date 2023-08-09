# # Tricks

#=
We demonstrate several features that may come in handy for some users.
=#

using ComponentArrays
using ForwardDiff
using ImplicitDifferentiation
using Krylov
using LinearAlgebra
using Random
using Test  #src
using Zygote

Random.seed!(63);

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

# You can use `ComponentVector` as an intermediate storage.

function forward_components(x::ComponentVector)
    d, e = forward_components_aux(x.a, x.b, x.m)
    y = ComponentVector(; d=d, e=e)
    return y
end

function conditions_components(x::ComponentVector, y::ComponentVector)
    c_d, c_e = conditions_components_aux(x.a, x.b, x.m, y.d, y.e)
    c = ComponentVector(; c_d=c_d, c_e=c_e)
    return c
end;

# And build your implicit function like so.

implicit_components = ImplicitFunction(forward_components, conditions_components);

# Since `ComponentVector`s are not yet compatible with iterative solvers from Krylov.jl, we (temporarily) need a bit of type piracy to make it work

Krylov.ktypeof(::ComponentVector{T,V}) where {T,V} = V

# Now we're good to go.

a, b, m = rand(2), rand(3), 7
x = ComponentVector(; a=a, b=b, m=m)
implicit_components(x)

# And it works with ForwardDiff.jl but not Zygote.jl (see documentation).

ForwardDiff.jacobian(implicit_components, x)
J = ForwardDiff.jacobian(forward_components, x)  #src
@test ForwardDiff.jacobian(implicit_components, x) ≈ J  #src
@test_broken Zygote.jacobian(implicit_components, x)[1] ≈ J  #src

#- The full differentiable pipeline looks like this

function full_pipeline(a, b, m)
    x = ComponentVector(; a=a, b=b, m=m)
    y = implicit_components(x)
    return y.d, y.e
end;

# ## Byproducts

# For when you need an additional output but don't care about its derivative.

function forward_byproduct(x)
    z = rand((2, 2))  # "randomized" choice
    y = x .^ (1 / z)
    return y, z
end

function conditions_byproduct(x, y, z)
    c = y .^ z .- x
    return c
end;

# The syntax for building the implicit function is the same.

implicit_byproduct = ImplicitFunction(forward_byproduct, conditions_byproduct);

# But this time the return value is a tuple `(y, z)`

x = rand(3)
implicit_byproduct(x)

# And it works with both ForwardDiff.jl and Zygote.jl

ForwardDiff.jacobian(first ∘ implicit_byproduct, x)
J = ForwardDiff.jacobian(first ∘ forward_byproduct, x)  #src
@test ForwardDiff.jacobian(first ∘ implicit_byproduct, x) ≈ J  #src

#-

Zygote.jacobian(first ∘ implicit_byproduct, x)[1]
@test Zygote.jacobian(first ∘ implicit_byproduct, x)[1] ≈ J  #src
