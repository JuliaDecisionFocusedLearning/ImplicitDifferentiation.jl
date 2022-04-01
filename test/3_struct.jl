# # Custom structs

#=
In this example, we demonstrate implicit differentiation through functions that manipulate `NamedTuple`s, as a first step towards compatibility with general structs.
=#

using ImplicitDifferentiation
using Krylov: gmres
using Zygote

using ChainRulesTestUtils  #src
using Test  #src

# ## Implicit function wrapper

#=
We replicate a componentwise square function with `NamedTuple`s:
=#

forward(a) = (u=a.x .^ 2, v=a.y .^ 2);

conditions(a, b) = vcat(b.u .- a.x .^ 2, b.v .- a.y .^ 2);

implicit = ImplicitFunction(; forward=forward, conditions=conditions, linear_solver=gmres);

#=
In order to be able to call `Zygote.gradient`, we use `implicit` to define a convoluted version of the squared Euclidean norm, which takes a vector as input and returns a real number.
=#

function mynorm(z::AbstractVector)
    n = length(z)
    x, y = z[1:(n ÷ 2)], z[(n ÷ 2 + 1):end]
    a = (x=x, y=y)
    b = implicit(a)
    return sum(b.u) + sum(b.v)
end;

# ## Testing

z = rand(5)
a = (x=z[1:2], y=z[3:5])  #src

# Let us first check that `mynorm` returns the correct result.

mynorm(z) ≈ sum(abs2, z)

# Now we go one step further and compute its gradient, which involves the reverse rule for `implicit`.

Zygote.gradient(mynorm, z)[1] ≈ 2z

# The following tests are not included in the docs.  #src

@test mynorm(z) ≈ sum(abs2, z)  #src
@test Zygote.gradient(mynorm, z)[1] ≈ 2z  #src

@testset verbose = true "ChainRules" begin  #src
    test_frule(implicit, a; check_inferred=false)  #src
    test_rrule(implicit, a; check_inferred=false)  #src
end  #src

using ChainRulesCore

_, pullback = rrule_via_ad(Zygote.ZygoteRuleConfig(), (b -> sum(b.v)) ∘ implicit, a)

pullback(2)
