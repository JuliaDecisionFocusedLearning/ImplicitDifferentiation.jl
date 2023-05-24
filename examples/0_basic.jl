# # Basic use

#=
In this example, we demonstrate the basics of our package on a simple function that is not amenable to automatic differentiation.
=#

using ChainRulesCore  #src
using ChainRulesTestUtils  #src
using ForwardDiff
using ImplicitDifferentiation
using JET  #src
using LinearAlgebra
using Random
using Test  #src
using Zygote

Random.seed!(63);

# ## Why do we bother?

#=
[ForwardDiff.jl](https://github.com/JuliaDiff/ForwardDiff.jl) and [Zygote.jl](https://github.com/FluxML/Zygote.jl) are two prominent packages for automatic differentiation in Julia.
While they are very generic, there are simple language constructs that they cannot differentiate through.
=#

function mysqrt(x::AbstractArray)
    a = [0.0]
    a[1] = first(x)
    return sqrt.(x)
end

#=
This is essentially the componentwise square root function but with an additional twist: `a::Vector{Float64}` is created internally, and its only element is replaced with the first element of `x`.
We can check that it does what it's supposed to do.
=#

x = rand(2)
mysqrt(x) ≈ sqrt.(x)
@test mysqrt(x) ≈ sqrt.(x)  #src

#=
Of course the Jacobian has an explicit formula.
=#

J = Diagonal(0.5 ./ sqrt.(x))

#=
However, things start to go wrong when we compute it with autodiff, due to the [limitations of ForwardDiff.jl](https://juliadiff.org/ForwardDiff.jl/stable/user/limitations/) and [those of Zygote.jl](https://fluxml.ai/Zygote.jl/stable/limitations/).
=#

try
    ForwardDiff.jacobian(mysqrt, x)
catch e
    e
end
@test_throws MethodError ForwardDiff.jacobian(mysqrt, x)  #src

#=
ForwardDiff.jl throws an error because it tries to call `mysqrt` with an array of dual numbers, and cannot use one of these numbers to fill `a` (which has element type `Float64`).
=#

try
    Zygote.jacobian(mysqrt, x)
catch e
    e
end
@test_throws ErrorException Zygote.jacobian(mysqrt, x)  #src

#=
Zygote.jl also throws an error because it cannot handle mutation.
=#

# ## Implicit function

#=
The first possible use of ImplicitDifferentiation.jl is to overcome the limitations of automatic differentiation packages by defining functions (and computing their derivatives) implicitly.
An implicit function is a mapping
```math
x \in \mathbb{R}^n \longmapsto y(x) \in \mathbb{R}^m
```
whose output is defined by conditions
```math
F(x,y(x)) = 0 \in \mathbb{R}^m
```
We represent it using a type called `ImplicitFunction`, which you will see in action shortly.
=#

#=
First we define a `forward` pass correponding to the function we consider.
It returns the actual output $y(x)$ of the function, as well as additional information $z(x)$.
Here we don't need any additional information, so we set it to $0$.
Importantly, this forward pass _doesn't need to be differentiable_.
=#

function forward(x)
    y = mysqrt(x)
    z = 0
    return y, z
end

#=
Then we define `conditions` $F(x, y, z) = 0$ that the output $y(x)$ is supposed to satisfy.
These conditions must be array-valued, with the same size as $y$, and take $z$ as an additional argument.
And unlike the forward pass, _the conditions need to be differentiable_ with respect to $x$ and $y$.
Here they are very obvious: the square of the square root should be equal to the original value.
=#

function conditions(x, y, z)
    c = y .^ 2 .- x
    return c
end

#=
Finally, we construct a wrapper `implicit` around the previous objects.
What does this wrapper do?
=#

implicit = ImplicitFunction(forward, conditions)

#=
When we call it as a function, it just falls back on `implicit.forward`, so unsurprisingly we get the same tuple $(y(x), z(x))$.
=#

(first ∘ implicit)(x) ≈ sqrt.(x)
@test (first ∘ implicit)(x) ≈ sqrt.(x)  #src

#=
And when we try to compute its Jacobian, the [implicit function theorem](https://en.wikipedia.org/wiki/Implicit_function_theorem) is applied in the background to circumvent the lack of differentiablility of the forward pass.
=#

# ## Forward and reverse mode autodiff

#=
Now ForwardDiff.jl works seamlessly.
=#

ForwardDiff.jacobian(first ∘ implicit, x) ≈ J
@test ForwardDiff.jacobian(first ∘ implicit, x) ≈ J  #src

#=
And so does Zygote.jl. Hurray!
=#

Zygote.jacobian(first ∘ implicit, x)[1] ≈ J
@test Zygote.jacobian(first ∘ implicit, x)[1] ≈ J  #src

# ## Second derivative

#=
We can even go higher-order by mixing the two packages (forward-over-reverse mode).
The only technical requirement is to switch the linear solver to something that can handle dual numbers:
=#

linear_solver(A, b) = (Matrix(A) \ b, (solved=true,))
implicit2 = ImplicitFunction(forward, conditions, linear_solver)

#=
Then the Jacobian itself is differentiable.
=#

h = rand(2)
J_Z(t) = Zygote.jacobian(first ∘ implicit2, x .+ t .* h)[1]
ForwardDiff.derivative(J_Z, 0) ≈ Diagonal((-0.25 .* h) ./ (x .^ 1.5))
@test ForwardDiff.derivative(J_Z, 0) ≈ Diagonal((-0.25 .* h) ./ (x .^ 1.5))  #src
