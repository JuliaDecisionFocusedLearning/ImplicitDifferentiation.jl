# # Tricks

#=
We demonstrate several features that may come in handy for some users.
=#

using ComponentArrays
using ForwardDiff
using ImplicitDifferentiation
using LinearAlgebra
using Optim
using Random
using Test  #src
using Zygote

Random.seed!(63);

# ## Multiple arguments

#=
First, we explain what to do when your forward mapping takes multiple input arguments:
```math
y(a, b) = \underset{y \in \mathbb{R}^m}{\mathrm{argmin}} ~ f(a, b, y)
```
The key idea is to store both $a$ and $b$ inside a single vector $x$.
=#

#=
To make verification easy, we minimize the following objective:
```math
f(a, b, y) = \lVert y \odot y - (a + 2b) \rVert^2
```
In this case, the optimization problem boils down to the componentwise square root of $a + 2b$.
We implement it with mutation to make sure that Zygote.jl will fail but ImplicitDifferentiation.jl will succeed.
=#

function mysqrt_components(a, b)
    y = copy(a)
    y .+= 2 .* b
    y .= sqrt.(y)
    return y
end

#=
First, we create the forward mapping which returns the solution $y(x)$, where $x$ is a `ComponentVector` containing both $a$ and $b$.
=#
function forward_components(x)
    return mysqrt_components(x.a, x.b)
end

#=
The optimality conditions are fairly easy to write.
=#

function conditions_components(x, y)
    return @. 2(y^2 - (x.a + 2x.b))
end

# We now have all the ingredients to construct our implicit function.

implicit_components = ImplicitFunction(forward_components, conditions_components)

# And remember, we should only call it on a very specific type of vector:

x = ComponentVector(; a=rand(2), b=rand(2))

#-

implicit_components(x) .^ 2
@test implicit_components(x) .^ 2 ≈ x.a + 2x.b  #src

#=
Let's see what the explicit Jacobian looks like.
=#

J = hcat(Diagonal(0.5 ./ sqrt.(x.a + 2x.b)), 2 * Diagonal(0.5 ./ sqrt.(x.a + 2x.b)))

# Forward mode autodiff

ForwardDiff.jacobian(implicit_components, x)
@test ForwardDiff.jacobian(implicit_components, x) ≈ J  #src

# Reverse mode autodiff

Zygote.jacobian(implicit_components, x)[1]
@test Zygote.jacobian(implicit_components, x)[1] ≈ J  #src

# ## Byproducts

#=
Next, we explain what to do when your forward mapping computes another object that you want to keep track of, which we will call its "byproduct".
The difference between this and multiple outputs (which should be managed with ComponentArrays.jl) is that _we do not compute derivatives with respect to byproducts_.
=#

#=
Imagine a situation where, depending on a coin toss, said mapping either doubles or halves the input.
After all, why not?
For each individual run, the algorithmic derivative is well-defined.
But to obtain it, you need to store the result of the toss. 
=#

function forward_cointoss(x)
    z = rand(Bool)
    if z
        y = 2x
    else
        y = x / 2
    end
    return y, z
end

#=
And naturally, the optimality condition also depends on the toss.
=#

function conditions_cointoss(x, y, z)
    if z
        return y .- 2x
    else
        return 2y .- x
    end
end

#=
The `ImplicitFunction` is created as usual:
=#

implicit_cointoss = ImplicitFunction(forward_cointoss, conditions_cointoss)

#=
But this time, when you call it, it will return a tuple:
=#

x = [1.0, 1.0]

implicit_cointoss(x)

#=
And as promised, differentiation works without taking the byproduct into account.
=#

Zygote.withjacobian(first ∘ implicit_cointoss, x)
