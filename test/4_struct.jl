# # Custom structs

#=
In this example, we demonstrate implicit differentiation through functions that manipulate `NamedTuple`s, as a first step towards compatibility with general structs.
=#

using ComponentArrays
using ImplicitDifferentiation
using Krylov: gmres
using Zygote

using ChainRulesCore  #src
using ChainRulesTestUtils  #src
using Test  #src

# ## Implicit function wrapper

#=
We replicate a componentwise square function with `NamedTuple`s, taking `a=(x,y)` as input and returning `b=(u,v)`.
=#

forward(a::ComponentVector) = ComponentVector(u=a.x .^ 2, v=a.y .^ 2);

function conditions(a::ComponentVector, b::ComponentVector)
    return vcat(b.u .- a.x .^ 2, b.v .- a.y .^ 2)
end

implicit = ImplicitFunction(; forward=forward, conditions=conditions, linear_solver=gmres);

#=
In order to be able to call `Zygote.gradient`, we use `implicit` to define a convoluted version of the squared Euclidean norm, which takes a `ComponentVector` as input and returns a real number.
=#

function fullnorm(a::ComponentVector)
    b = implicit(a)
    return sum(b)
end

function halfnorm(a::ComponentVector)  #src
    b = implicit(a)  #src
    return sum(b.u)  #src
end;  #src

# ## Testing

a = ComponentVector(x=rand(3), y=rand(3))

# Let us first check that our weird squared norm returns the correct result.

fullnorm(a) ≈ sum(abs2, a)
halfnorm(a) ≈ sum(abs2, a.x)  #src

# Now we go one step further and compute its gradient, which involves the reverse rule for `implicit`.

Zygote.gradient(fullnorm, a)[1] ≈ 2a

#-

Zygote.gradient(halfnorm, a)[1] ≈ vcat(2a.x, zero(a.y))  #src

# The following tests are not included in the docs.  #src

@test fullnorm(a) ≈ sum(abs2, a)  #src
@test halfnorm(a) ≈ sum(abs2, a.x)  #src

@test Zygote.gradient(fullnorm, a)[1] ≈ 2a  #src
@test Zygote.gradient(halfnorm, a)[1] ≈ vcat(2a.x, zero(a.y))  #src
