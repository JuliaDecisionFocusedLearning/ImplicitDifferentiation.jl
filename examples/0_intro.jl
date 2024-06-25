# # Introduction

#=
We explain the basics of our package on a simple function that is not amenable to naive automatic differentiation.
=#

using ForwardDiff
using ImplicitDifferentiation
using JET  #src
using LinearAlgebra
using Test  #src
using Zygote

# ## Why do we bother?

#=
[ForwardDiff.jl](https://github.com/JuliaDiff/ForwardDiff.jl) and [Zygote.jl](https://github.com/FluxML/Zygote.jl) are two prominent packages for automatic differentiation in Julia.
While they are very generic, there are simple language constructs that they cannot differentiate through.
=#

function badsqrt(x::AbstractArray)
    a = [0.0]
    a[1] = x[1]
    return sqrt.(x)
end;

#=
This is essentially the componentwise square root function but with an additional twist: `a::Vector{Float64}` is created internally, and its only element is replaced with the first element of `x`.
We can check that it does what it's supposed to do.
=#

x = [4.0, 9.0]
badsqrt(x)
@test badsqrt(x) ≈ sqrt.(x)  #src

#=
Of course the Jacobian has an explicit formula.
=#

J = Diagonal(0.5 ./ sqrt.(x))

#=
However, things start to go wrong when we compute it with autodiff, due to the [limitations of ForwardDiff.jl](https://juliadiff.org/ForwardDiff.jl/stable/user/limitations/) and [those of Zygote.jl](https://fluxml.ai/Zygote.jl/stable/limitations/).
=#

try
    ForwardDiff.jacobian(badsqrt, x)
catch e
    e
end
@test_throws MethodError ForwardDiff.jacobian(badsqrt, x)  #src

#=
ForwardDiff.jl throws an error because it tries to call `badsqrt` with an array of dual numbers, and cannot use one of these numbers to fill `a` (which has element type `Float64`).
=#

try
    Zygote.jacobian(badsqrt, x)
catch e
    e
end
@test_throws ErrorException Zygote.jacobian(badsqrt, x)  #src

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
c(x,y(x)) = 0 \in \mathbb{R}^m
```
We represent it using a type called [`ImplicitFunction`](@ref), which you will see in action shortly.
=#

#=
First we define a `forward` mapping corresponding to the function we consider.
It returns the actual output $y(x)$ of the function, and can be thought of as a black box solver.
Importantly, this Julia callable doesn't need to be differentiable by automatic differentiation packages but the underlying function still needs to be mathematically differentiable.
=#

forward(x) = badsqrt(x);

#=
Then we define `conditions` $c(x, y) = 0$ that the output $y(x)$ is supposed to satisfy.
These conditions must be array-valued, with the same size as $y$.
Unlike the forward mapping, the conditions need to be differentiable by automatic differentiation packages with respect to both $x$ and $y$.
Here the conditions are very obvious: the square of the square root should be equal to the original value.
=#

function conditions(x, y)
    c = y .^ 2 .- x
    return c
end;

#=
Finally, we construct a wrapper `implicit` around the previous objects.
By default, `forward` is assumed to return a single output and `conditions`
is assumed to accept 2 arguments.
=#

implicit = ImplicitFunction(forward, conditions)

#=
What does this wrapper do?
When we call it as a function, it just falls back on `implicit.forward`, so unsurprisingly we get the output $y(x)$.
=#

implicit(x)
@test implicit(x) ≈ sqrt.(x)  #src

#=
And when we try to compute its Jacobian, the [implicit function theorem](https://en.wikipedia.org/wiki/Implicit_function_theorem) is applied in the background to circumvent the lack of differentiability of the forward mapping.
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
