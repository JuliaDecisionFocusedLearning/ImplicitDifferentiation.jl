# # Basic use

#=
In this example, we demonstrate the basics of our package on a simple function that is not amenable to automatic differentiation.
=#

using ChainRulesCore  #src
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
First we define a forward mapping correponding to the function we consider.
It returns the actual output $y(x)$ of the function, and can be thought of as a black box solver.
Importantly, this Julia callable _doesn't need to be differentiable by automatic differentiation packages but the underlying function still needs to be mathematically differentiable_.
=#

forward(x) = mysqrt(x)

#=
Optionally, the forward mapping can also return some additional byproduct $z$, e.g a pre-computed Jacobian, which would then be used in the conditions.
=#

forward2(x) = (mysqrt(x), 0)

#=
Then we define `conditions` $c(x, y) = 0$ that the output $y(x)$ is supposed to satisfy.
These conditions must be array-valued, with the same size as $y$.
If the forward mapping returns an additional byproduct $z$, the conditions function must also accept a third argument $z$, such that $c(x, y, z) = 0$. Unlike the forward mapping, _the conditions need to be differentiable by automatic differentiation packages_ with respect to both $x$ and $y$.
Here the conditions are very obvious: the square of the square root should be equal to the original value.
=#

function conditions(x, y)
    c = y .^ 2 .- x
    return c
end

#=
Or if we add a third argument:
=#

function conditions2(x, y, z)
    c = y .^ 2 .- x
    return c
end

#=
Finally, we construct a wrapper `implicit` around the previous objects.
By default, `forward` is assumed to return a single output and `conditions`
is assumed to accept 2 arguments.
=#

implicit = ImplicitFunction(forward, conditions)

#=
Alternatively, we can use `forward2` which returns two outputs and
`conditions2 ` which accepts three arguments.
In this case, we must pass `Val(true)` to the `ImplicitFunction` constructor, so that it knows we have a byproduct to carry around.
=#

implicit2 = ImplicitFunction(forward2, conditions2, Val(true))

#=
What does this wrapper do?
When we call it as a function, it just falls back on `first ∘ implicit.forward`, so unsurprisingly we get the first output $y(x)$.
=#

implicit(x) ≈ sqrt.(x)
@test implicit(x) ≈ sqrt.(x)  #src

#=
And when we try to compute its Jacobian, the [implicit function theorem](https://en.wikipedia.org/wiki/Implicit_function_theorem) is applied in the background to circumvent the lack of differentiablility of the forward pass.
=#

# ## Forward and reverse mode autodiff

#=
Now ForwardDiff.jl works seamlessly.
=#

ForwardDiff.jacobian(implicit, x) ≈ J
@test ForwardDiff.jacobian(implicit, x) ≈ J  #src

#=
And so does Zygote.jl. Hurray!
=#

Zygote.jacobian(implicit, x)[1] ≈ J
@test Zygote.jacobian(implicit, x)[1] ≈ J  #src

# ## Second derivative

#=
We can even go higher-order by mixing the two packages (forward-over-reverse mode).
The only technical requirement is to switch the linear solver to something that can handle dual numbers:
=#

manual_linear_solver(A, b) = (Matrix(A) \ b, (solved=true,))

implicit_higher_order = ImplicitFunction(
    forward, conditions; linear_solver=manual_linear_solver
)

#=
Then the Jacobian itself is differentiable.
=#

h = rand(2)
J_Z(t) = Zygote.jacobian(implicit_higher_order, x .+ t .* h)[1]
ForwardDiff.derivative(J_Z, 0) ≈ Diagonal((-0.25 .* h) ./ (x .^ 1.5))
@test ForwardDiff.derivative(J_Z, 0) ≈ Diagonal((-0.25 .* h) ./ (x .^ 1.5))  #src
