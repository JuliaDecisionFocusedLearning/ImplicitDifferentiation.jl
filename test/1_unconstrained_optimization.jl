# # Unconstrained optimization

#=
In this example, we show how to differentiate through the solution of the following unconstrained optimization problem:
```math
\hat{y}(x) = \min_{y \in \mathbb{R}^m} f(x, y)
```
The optimality conditions are given by gradient stationarity:
```math
F(x, \hat{y}(x)) = 0 \quad \text{with} \quad F(x,y) = \nabla_2 f(x, y) = 0
```

=#

using ImplicitDifferentiation
using Krylov: gmres
using Optim: optimize, minimizer, LBFGS
using Zygote

using ChainRulesTestUtils  #src
using Test  #src

# ## Implicit function wrapper

#=
To make verification easy, we minimize a quadratic objective
```math
f(x, y) = \lVert y - x \rVert^2
```
In this case, the optimization algorithm is very simple, but still we can implement it as a black box to show that it doesn't change the result.
=#

function forward(x)
    f(y) = sum(abs2, y-x)
    y0 = zero(x)
    res = optimize(f, y0, LBFGS(); autodiff=:forward)
    y = minimizer(res)
    return y, nothing
end;

#=
On the other hand, optimality conditions should be provided explicitly whenever possible, so as to avoid nesting automatic differentiation calls.
=#

conditions(x, y, useful_info=nothing) = 2(y - x);

# We now have all the ingredients to construct our implicit function.

implicit = ImplicitFunction(; forward=forward, conditions=conditions, linear_solver=gmres);

# ## Testing

x = rand(5)

# Let's start by taking a look at the forward pass, which should be the identity function.

implicit(x)

# We now check whether the behavior of our `ImplicitFunction` wrapper is coherent with the theoretical derivatives.

Zygote.jacobian(implicit, x)[1]

# As expected, we recover the identity matrix as Jacobian.

# The following tests are not included in the docs.  #src

@testset verbose = true "ChainRules" begin  #src
    test_frule(implicit, x; check_inferred=false)  #src
    test_rrule(implicit, x; check_inferred=false)  #src
end  #src
