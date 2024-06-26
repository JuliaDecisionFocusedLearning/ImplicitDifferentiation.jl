# # Basic use cases

#=
We show how to differentiate through very common routines:
- an unconstrained optimization problem
- a nonlinear system of equations
- a fixed point iteration

Note that some packages from the [SciML](https://sciml.ai/) ecosystem provide a similar implicit differentiation mechanism.
=#

using ForwardDiff
using ImplicitDifferentiation
using LinearAlgebra
using NLsolve
using Optim
using Test  #src
using Zygote

#=
In all three cases, we will use the square root as our forward mapping, but expressed in three different ways.
Here's our heroic test vector:
=#

x = [4.0, 9.0];

#=
Since we already know the mathematical expression of the Jacobian, we will be able to compare it with our numerical results.
=#

J = Diagonal(0.5 ./ sqrt.(x))

# ## Unconstrained optimization

#=
First, we show how to differentiate through the solution of an unconstrained optimization problem:
```math
y(x) = \underset{y \in \mathbb{R}^m}{\mathrm{argmin}} ~ f(x, y)
```
The optimality conditions are given by gradient stationarity:
```math
c(x, y) = \nabla_2 f(x, y) = 0
```
=#

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
end;

#=
Even though they are defined as a gradient, it is better to provide optimality conditions explicitly: that way we avoid nesting autodiff calls. By default, the conditions should accept two arguments as input.
The forward mapping and the conditions should accept the same set of keyword arguments.
=#

function conditions_optim(x, y; method)
    ∇₂f = @. 4 * (y^2 - x) * y
    return ∇₂f
end;

#=
We now have all the ingredients to construct our implicit function.
=#

implicit_optim = ImplicitFunction(forward_optim, conditions_optim)

# And indeed, it behaves as it should when we call it:

implicit_optim(x; method=LBFGS()) .^ 2
@test implicit_optim(x; method=LBFGS()) .^ 2 ≈ x  #src

# Forward mode autodiff

ForwardDiff.jacobian(_x -> implicit_optim(_x; method=LBFGS()), x)
@test ForwardDiff.jacobian(_x -> implicit_optim(_x; method=LBFGS()), x) ≈ J  #src

#=
In this instance, we could use ForwardDiff.jl directly on the solver:
=#

ForwardDiff.jacobian(_x -> forward_optim(_x; method=LBFGS()), x)

# Reverse mode autodiff

Zygote.jacobian(_x -> implicit_optim(_x; method=LBFGS()), x)[1]
@test Zygote.jacobian(_x -> implicit_optim(_x; method=LBFGS()), x)[1] ≈ J  #src

#=
In this instance, we cannot use Zygote.jl directly on the solver (due to unsupported `try/catch` statements).
=#

try
    Zygote.jacobian(_x -> forward_optim(_x; method=LBFGS()), x)[1]
catch e
    e
end

# ## Nonlinear system

#=
Next, we show how to differentiate through the solution of a nonlinear system of equations:
```math
\text{find} \quad y(x) \quad \text{such that} \quad c(x, y(x)) = 0
```
The optimality conditions are pretty obvious:
```math
c(x, y) = 0
```
=#

#=
To make verification easy, we solve the following system:
```math
c(x, y) = y \odot y - x = 0
```
In this case, the optimization problem boils down to the componentwise square root function, but we implement it using a black box solver from [NLsolve.jl](https://github.com/JuliaNLSolvers/NLsolve.jl).
=#

function forward_nlsolve(x; method)
    F!(storage, y) = (storage .= y .^ 2 .- x)
    initial_y = similar(x)
    initial_y .= 1
    result = nlsolve(F!, initial_y; method)
    return result.zero
end;

#-

function conditions_nlsolve(x, y; method)
    c = y .^ 2 .- x
    return c
end;

#-

implicit_nlsolve = ImplicitFunction(forward_nlsolve, conditions_nlsolve)

#-

implicit_nlsolve(x; method=:newton) .^ 2
@test implicit_nlsolve(x; method=:newton) .^ 2 ≈ x  #src

# Forward mode autodiff

ForwardDiff.jacobian(_x -> implicit_nlsolve(_x; method=:newton), x)
@test ForwardDiff.jacobian(_x -> implicit_nlsolve(_x; method=:newton), x) ≈ J  #src

#-

ForwardDiff.jacobian(_x -> forward_nlsolve(_x; method=:newton), x)

# Reverse mode autodiff

Zygote.jacobian(_x -> implicit_nlsolve(_x; method=:newton), x)[1]
@test Zygote.jacobian(_x -> implicit_nlsolve(_x; method=:newton), x)[1] ≈ J  #src

#-

try
    Zygote.jacobian(_x -> forward_nlsolve(_x; method=:newton), x)[1]
catch e
    e
end

# ## Fixed point

#=
Finally, we show how to differentiate through the limit of a fixed point iteration:
```math
y \longmapsto g(x, y)
```
The optimality conditions are pretty obvious:
```math
c(x, y) = g(x, y) - y = 0
```
=#

#=
To make verification easy, we consider [Heron's method](https://en.wikipedia.org/wiki/Methods_of_computing_square_roots#Heron's_method):
```math
g(x, y) = \frac{1}{2} \left(y + \frac{x}{y}\right)
```
In this case, the fixed point algorithm boils down to the componentwise square root function, but we implement it manually.
=#

function forward_fixedpoint(x; iterations)
    y = ones(eltype(x), size(x))
    for _ in 1:iterations
        y .= (y .+ x ./ y) ./ 2
    end
    return y
end;

#-

function conditions_fixedpoint(x, y; iterations)
    g = (y .+ x ./ y) ./ 2
    return g .- y
end;

#-

implicit_fixedpoint = ImplicitFunction(forward_fixedpoint, conditions_fixedpoint)

#-

implicit_fixedpoint(x; iterations=10) .^ 2
@test implicit_fixedpoint(x; iterations=10) .^ 2 ≈ x  #src

# Forward mode autodiff

ForwardDiff.jacobian(_x -> implicit_fixedpoint(_x; iterations=10), x)
@test ForwardDiff.jacobian(_x -> implicit_fixedpoint(_x; iterations=10), x) ≈ J  #src

#-

ForwardDiff.jacobian(_x -> forward_fixedpoint(_x; iterations=10), x)

# Reverse mode autodiff

Zygote.jacobian(_x -> implicit_fixedpoint(_x; iterations=10), x)[1]
@test Zygote.jacobian(_x -> implicit_fixedpoint(_x; iterations=10), x)[1] ≈ J  #src

#-

try
    Zygote.jacobian(_x -> forward_fixedpoint(_x; iterations=10), x)[1]
catch e
    e
end
