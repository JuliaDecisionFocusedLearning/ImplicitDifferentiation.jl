# # Fixed point

#=
In this example, we show how to differentiate through the limit of a fixed point iteration:
```math
y \longmapsto T(x, y)
```
The optimality conditions are pretty obvious:
```math
y = T(x, y)
```
=#

using ForwardDiff
using ImplicitDifferentiation
using LinearAlgebra
using Random
using Test  #src
using Zygote

Random.seed!(63);

# ## Implicit function

#=
To make verification easy, we consider [Heron's method](https://en.wikipedia.org/wiki/Methods_of_computing_square_roots#Heron's_method):
```math
T(x, y) = \frac{1}{2} \left(y + \frac{x}{y}\right)
```
In this case, the fixed point algorithm boils down to the componentwise square root function, but we implement it manually.
=#

function mysqrt_fixedpoint(x; iterations)
    y = ones(eltype(x), size(x))
    for _ in 1:iterations
        y .= 0.5 .* (y .+ x ./ y)
    end
    return y
end

#-

function forward_fixedpoint(x; iterations)
    y = mysqrt_fixedpoint(x; iterations)
    z = 0
    return y, z
end

#-

function conditions_fixedpoint(x, y, z; iterations)
    T = 0.5 .* (y .+ x ./ y)
    return T .- y
end

#-

implicit_fixedpoint = ImplicitFunction(forward_fixedpoint, conditions_fixedpoint)

#-

x = rand(2)

#-

(first ∘ implicit_fixedpoint)(x; iterations=10) .^ 2
@test (first ∘ implicit_fixedpoint)(x; iterations=10) .^ 2 ≈ x  #src

#-

J = Diagonal(0.5 ./ sqrt.(x))

# ## Forward mode autodiff

ForwardDiff.jacobian(_x -> first(implicit_fixedpoint(_x; iterations=10)), x)
@test ForwardDiff.jacobian(_x -> first(implicit_fixedpoint(_x; iterations=10)), x) ≈ J  #src

#-

ForwardDiff.jacobian(_x -> mysqrt_fixedpoint(_x; iterations=10), x)

# ## Reverse mode autodiff

Zygote.jacobian(_x -> first(implicit_fixedpoint(_x; iterations=10)), x)[1]
@test Zygote.jacobian(_x -> first(implicit_fixedpoint(_x; iterations=10)), x)[1] ≈ J  #src

#-

try
    Zygote.jacobian(_x -> mysqrt_fixedpoint(_x; iterations=10), x)[1]
catch e
    e
end
