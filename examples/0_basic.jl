# # Basic use

#=
In this example, we demonstrate the basics of our package on a simple function that is not amenable to automatic differentiation.
=#

using ForwardDiff
using ImplicitDifferentiation
using LinearAlgebra
using Random
using Test  #src
using Zygote

Random.seed!(63);

# ## Autodiff fails

#=
ForwardDiff.jl and Zygote.jl are two prominent packages for automatic differentiation in Julia.
While they are very generic, there are simple language constructs that they cannot differentiate through.
=#

function mysquare(x::AbstractArray)
    a = [0.0]
    a[1] = first(x)
    return x .^ 2
end;

#=
This is basically the componentwise square function but with an additional twist: `a::Vector{Float64}` is created internally, and its only element is replaced with the first element of `x`.
We can check that it does what it's supposed to do.
=#

x = rand(2)
mysquare(x) ≈ x .^ 2
@test mysquare(x) ≈ x .^ 2  #src

#=
However, things start to go wrong when we compute Jacobians, due to the limitations of [ForwardDiff.jl](https://juliadiff.org/ForwardDiff.jl/stable/user/limitations/) and [Zygote.jl](https://fluxml.ai/Zygote.jl/stable/limitations/).
=#

try
    ForwardDiff.jacobian(mysquare, x)
catch e
    e
end
@test_throws MethodError ForwardDiff.jacobian(mysquare, x)  #src

#=
ForwardDiff.jl throws an error because it tries to call `mysquare` with an array of dual numbers, and cannot use one of these numbers to fill `a` (which has element type `Float64`).
=#

try
    Zygote.jacobian(mysquare, x)
catch e
    e
end
@test_throws ErrorException Zygote.jacobian(mysquare, x)  #src

#=
Zygote.jl also throws an error because it cannot handle mutation.
=#

# ## Implicit functions

#=
The first possible use of ImplicitDifferentiation.jl is to overcome the limitations of automatic differentiation packages by defining Jacobians implicitly.
Its main export is a type called [`ImplicitFunction`](@ref), which we are going to see in action.
=#

#=
First we define a `forward` pass correponding to the function we consider.
It returns the actual output `y` of the function, as well as additional information `z` (which we don't need here, hence the `0`).
Importantly, this forward pass _doesn't need to be differentiable_.
=#

function forward(x)
    y = mysquare(x)
    z = 0
    return y, z
end

#=
Then we define `conditions` that the output `y` is supposed to satisfy.
These conditions must be array-valued, with the same size as `y`.
Here they are very obvious, but in later examples they will be more involved.
=#

function conditions(x, y, z)
    c = y .- (x .^ 2)
    return c
end

#=
Finally, we construct a wrapper `implicit` around the previous objects.
What does this wrapper do?
=#

implicit = ImplicitFunction(forward, conditions)

#=
When we call it as a function, it just falls back on `implicit.forward`, so unsurprisingly we get the same tuple `(y, z)`.
=#

implicit(x)[1] ≈ x .^ 2
@test implicit(x)[1] ≈ x .^ 2  #src

# ## Autodiff works

#=
And when we try to compute its Jacobian, the [implicit function theorem](https://en.wikipedia.org/wiki/Implicit_function_theorem) is applied in the background to circumvent the lack of differentiablility of the forward pass.
=#

#=
Now ForwardDiff.jl works seamlessly.
=#

ForwardDiff.jacobian(first ∘ implicit, x) ≈ Diagonal(2x)
@test ForwardDiff.jacobian(first ∘ implicit, x) ≈ Diagonal(2x)  #src

#=
And so does Zygote.jl. Hurray!
=#

Zygote.jacobian(first ∘ implicit, x)[1] ≈ Diagonal(2x)
@test Zygote.jacobian(first ∘ implicit, x)[1] ≈ Diagonal(2x)  #src

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
ForwardDiff.derivative(J_Z, 0) ≈ Diagonal(2h)
@test ForwardDiff.derivative(J_Z, 0) ≈ Diagonal(2h)  #src
