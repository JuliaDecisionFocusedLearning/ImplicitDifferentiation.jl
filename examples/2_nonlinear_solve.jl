# # Nonlinear solve

#=
In this example, we show how to differentiate through the solution of a nonlinear system of equations:
```math
\text{find} \quad y(x) \quad \text{such that} \quad F(x, y(x)) = 0
```
The optimality conditions are pretty obvious:
```math
F(x, y) = 0
```
=#

using ForwardDiff
using ImplicitDifferentiation
using LinearAlgebra
using NLsolve
using Random
using Test  #src
using Zygote

Random.seed!(63);

# ## Implicit function

#=
To make verification easy, we solve the following system:
```math
F(x, y) = y \odot y - x = 0
```
In this case, the optimization problem boils down to the componentwise square root function, but we implement it using a black box solver from [NLsolve.jl](https://github.com/JuliaNLSolvers/NLsolve.jl).
=#

function mysqrt_nlsolve(x; method)
    F!(storage, y) = (storage .= y .^ 2 - x)
    initial_y = ones(eltype(x), size(x))
    result = nlsolve(F!, initial_y; method)
    return result.zero
end

#-

function forward_nlsolve(x; method)
    y = mysqrt_nlsolve(x; method)
    z = 0
    return y, z
end

#-

function conditions_nlsolve(x, y, z; method)
    F = y .^ 2 .- x
    return F
end

#-

implicit_nlsolve = ImplicitFunction(forward_nlsolve, conditions_nlsolve)

#-

x = rand(2)

#-

implicit_nlsolve(x; method=:newton) .^ 2
@test implicit_nlsolve(x; method=:newton) .^ 2 ≈ x  #src

#-

J = Diagonal(0.5 ./ sqrt.(x))

# ## Forward mode autodiff

ForwardDiff.jacobian(_x -> implicit_nlsolve(_x; method=:newton), x)
@test ForwardDiff.jacobian(_x -> implicit_nlsolve(_x; method=:newton), x) ≈ J  #src

#-

ForwardDiff.jacobian(_x -> mysqrt_nlsolve(_x; method=:newton), x)

# ## Reverse mode autodiff

Zygote.jacobian(_x -> implicit_nlsolve(_x; method=:newton), x)[1]
@test Zygote.jacobian(_x -> implicit_nlsolve(_x; method=:newton), x)[1] ≈ J  #src

#-

try
    Zygote.jacobian(_x -> mysqrt_nlsolve(_x; method=:newton), x)[1]
catch e
    e
end
