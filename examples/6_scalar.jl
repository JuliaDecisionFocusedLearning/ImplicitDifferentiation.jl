# # Scalar output

#=
In this example, we check that everything still works if the function returns a scalar output instead of an array.
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
The task is to compute the sum of the the componentwise square root of $x$.
=#

function mysqrt_sum(x)
    y = sqrt(sum(x))
    return y
end

#-

function forward_sum(x)
    y = mysqrt_sum(x)
    z = 0
    return y, z
end

#-

function conditions_sum(x, y, z)
    return y^2 - sum(x)
end

#-

implicit_sum = ImplicitFunction(forward_sum, conditions_sum)

#-

x = rand(2)

#-

first(implicit_sum(x)) ^ 2
@test first(implicit_sum(x)) ^ 2 ≈ sum(x)  #src

#=
Let's see what the explicit gradient looks like.
=#

g = [0.5 / sqrt(sum(x)) for _ in eachindex(x)]

# ## Forward mode autodiff

ForwardDiff.gradient(_x -> first(implicit_sum(_x)), x)
@test ForwardDiff.gradient(_x -> first(implicit_sum(_x)), x) ≈ g  #src

# ## Reverse mode autodiff

@test Zygote.jacobian(_x -> first(implicit_sum(_x)), x)[1] ≈ J  #src
