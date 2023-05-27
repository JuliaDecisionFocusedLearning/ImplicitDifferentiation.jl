# # Multiple arguments

#=
In this example, we explain what to do when your function takes multiple input arguments:
```math
y(a, b) = \underset{y \in \mathbb{R}^m}{\mathrm{argmin}} ~ f(a, b, y)
```
The key idea is to store both $a$ and $b$ inside a single vector $x$.
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

# ## Implicit function

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
First, we create the forward pass which returns the solution $y(x)$, where $x$ is a `ComponentVector` containing both $a$ and $b$.
=#
function forward_components(x)
    y = mysqrt_components(x.a, x.b)
    z = 0
    return y, z
end

#=
The optimality conditions are fairly easy to write.
=#

function conditions_components(x, y, z)
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

# ## Forward mode autodiff

ForwardDiff.jacobian(implicit_components, x)
@test ForwardDiff.jacobian(implicit_components, x) ≈ J  #src

# ## Reverse mode autodiff

Zygote.jacobian(implicit_components, x)[1]
@test Zygote.jacobian(implicit_components, x)[1] ≈ J  #src
