# # Unconstrained optimization

#=
In this example, we show how to differentiate through the solution of an unconstrained optimization problem:
```math
y(x) = \underset{y \in \mathbb{R}^m}{\mathrm{argmin}} ~ f(x, y)
```
The optimality conditions are given by gradient stationarity:
```math
\nabla_2 f(x, y) = 0
```
=#

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
f(x, y) = \lVert y \odot y - x \rVert^2
```
In this case, the optimization problem boils down to the componentwise square root function, but we implement it using a black box solver from [Optim.jl](https://github.com/JuliaNLSolvers/Optim.jl).
Note the presence of a keyword argument.
=#

function forward_optim(x; method)
    f(y) = sum(abs2, y .^ 2 .- x)
    y0 = ones(eltype(x), size(x))
    result = optimize(f, y0, method)
    return Optim.minimizer(result)
end

#=
Even though they are defined as a gradient, it is better to provide optimality conditions explicitly: that way we avoid nesting autodiff calls. By default, the conditions should accept two arguments as input.
The forward mapping and the conditions should accept the same set of keyword arguments.
=#

function conditions_optim(x, y; method)
    ∇₂f = 2 .* (y .^ 2 .- x)
    return ∇₂f
end

#=
We now have all the ingredients to construct our implicit function.
=#

implicit_optim = ImplicitFunction(forward_optim, conditions_optim)

# And indeed, it behaves as it should when we call it:

x = rand(2)

#-

implicit_optim(x; method=LBFGS()) .^ 2
@test implicit_optim(x; method=LBFGS()) .^ 2 ≈ x  #src

#=
Let's see what the explicit Jacobian looks like.
=#

J = Diagonal(0.5 ./ sqrt.(x))

# ## Forward mode autodiff

ForwardDiff.jacobian(_x -> implicit_optim(_x; method=LBFGS()), x)
@test ForwardDiff.jacobian(_x -> implicit_optim(_x; method=LBFGS()), x) ≈ J  #src

#=
Unsurprisingly, the Jacobian is the identity.
In this instance, we could use ForwardDiff.jl directly on the solver, but it returns the wrong result (not sure why).
=#

ForwardDiff.jacobian(_x -> forward_optim(x; method=LBFGS()), x)

# ## Reverse mode autodiff

Zygote.jacobian(_x -> implicit_optim(_x; method=LBFGS()), x)[1]
@test Zygote.jacobian(_x -> implicit_optim(_x; method=LBFGS()), x)[1] ≈ J  #src

#=
Again, the Jacobian is the identity.
In this instance, we cannot use Zygote.jl directly on the solver (due to unsupported `try/catch` statements).
=#

try
    Zygote.jacobian(_x -> forward_optim(x; method=LBFGS()), x)[1]
catch e
    e
end
