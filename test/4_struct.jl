# # Custom structs

#=
In this example, we demonstrate implicit differentiation through functions that manipulate `NamedTuple`s, as a first step towards compatibility with general structs.
=#

using ImplicitDifferentiation
using Krylov: gmres
using Zygote

using Diffractor  #src
using ChainRulesCore  #src
using ChainRulesTestUtils  #src
using Test  #src

# ## Implicit function wrapper

#=
We replicate a componentwise square function with `NamedTuple`s, taking `a=(x,y)` as input and returning `b=(u,v)`.
=#

forward(a) = (u=a.x .^ 2, v=a.y .^ 2);

function conditions(a, b)
    return vcat(b.u .- a.x .^ 2, b.v .- a.y .^ 2)
end

implicit = ImplicitFunction(; forward=forward, conditions=conditions, linear_solver=gmres);

#=
In order to be able to call `Zygote.gradient`, we use `implicit` to define a convoluted version of the squared Euclidean norm, which takes a vector as input and returns a real number.
=#

function mynorm(z::AbstractVector)
    n = length(z)
    a = (x=z[1:n÷2], y=z[n÷2+1:end])
    b = implicit(a)
    return sum(b.u) + sum(b.v)
end;

function mynorm_broken(x::AbstractVector)  #src
    a = (x=x, y=rand(2))  #src
    b = implicit(a)  #src
    return sum(b.u)  #src
end;  #src

# ## Testing

x = rand(5)

# Let us first check that our weird squared norm returns the correct result.

mynorm(x) ≈ sum(abs2, x)

# Now we go one step further and compute its gradient, which involves the reverse rule for `implicit`.

Zygote.gradient(mynorm, x)[1] ≈ 2x

# The following tests are not included in the docs.  #src

@test mynorm(x) ≈ sum(abs2, x)  #src
@test mynorm_broken(x) ≈ sum(abs2, x)  #src

@test Zygote.gradient(mynorm, x)[1] ≈ 2x  #src
@test_broken Zygote.gradient(mynorm_broken, x)[1] ≈ 2x  #src

# EXPERIMENTAL  #src

zrc = Zygote.ZygoteRuleConfig()  #src
_, pullback = rrule_via_ad(zrc, mynorm, x)  #src
_, pullback_broken = rrule_via_ad(zrc, mynorm_broken, x)  #src
pullback(1)  #src
pullback_broken(1)  #src
