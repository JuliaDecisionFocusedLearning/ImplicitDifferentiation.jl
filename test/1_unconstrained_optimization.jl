# # Unconstrained optimization

#=
In this example, we show how to differentiate through the solution of the following unconstrained optimization problem:
```math
\hat{y}(x) = \min_{y \in \mathbb{R}^m} f(x, y)
```
The optimality conditions are given by gradient stationarity:
```math
F(x, \hat{y}(x)) = \nabla_1 f(x, \hat{y}(x)) = 0
```

=#

using GalacticOptim
using ImplicitDifferentiation
using IterativeSolvers
using LinearAlgebra
using Optim
using Statistics
using Zygote

using ChainRulesTestUtils  #src
using Test  #src

# ## White box

#=
To make verification easy, we minimize a quadratic objective
```math
f(x, y) = \lVert x - y \rVert^2
```
In this case, the optimization algorithm and optimality conditions have very simple expressions.
=#

forward_white_box(x) = x
conditions_white_box(x, y) = 2(x - y)

implicit_white_box = ImplicitFunction(; forward=forward_white_box, conditions=conditions_white_box, linear_solver=gmres)

# ## Black box

#=
However, we can also handle the case where no formulae are known, and the user must resort to black box optimization and differentiation algorithms.
=#

square_distance(x, y) = sum(abs2, x - y)

function forward_black_box(x)
    fun = OptimizationFunction(square_distance, GalacticOptim.AutoForwardDiff())
    prob = OptimizationProblem(fun, zero(x), x)
    sol = GalacticOptim.solve(prob, LBFGS())
    return sol.u
end

function conditions_black_box(x, y)
    gs = Zygote.gradient(ỹ -> square_distance(ỹ, x), y)
    return gs[1]
end

implicit_black_box = ImplicitFunction(;
    forward=forward_black_box, conditions=conditions_black_box, linear_solver=gmres
)

# ## Testing

x = rand(5)

# We now check whether the behavior of our `ImplicitFunction` wrapper is coherent with the theoretical derivatives.

Zygote.jacobian(implicit_white_box, x)[1]

#

Zygote.jacobian(implicit_black_box, x)[1]

# As expected, we recover the identity matrix as Jacobian.

# The following tests are not included in the docs.  #src

@testset verbose = true "White box" begin  #src
    test_frule(implicit_white_box, x)  #src
    test_rrule(implicit_white_box, x)  #src
end  #src

@testset verbose = true "Black box" begin  #src
    test_frule(implicit_black_box, x)  #src
    test_rrule(implicit_black_box, x)  #src
end  #src
