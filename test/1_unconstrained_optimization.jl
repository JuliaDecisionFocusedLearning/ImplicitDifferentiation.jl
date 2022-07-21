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
using Optim
using Random
using Zygote

using ChainRulesTestUtils  #src
using Test  #src

Random.seed!(63)

# ## Implicit function wrapper

#=
To make verification easy, we minimize a quadratic objective
```math
f(x, y) = \lVert y - x \rVert^2
```
In this case, the optimization algorithm is very simple (the identity function does the job), but still we implement it using a black box solver from `Optim.jl` to show that it doesn't change the result.
=#

function dumb_identity(x)
    f(y) = sum(abs2, y-x)
    y0 = zero(x)
    res = optimize(f, y0, LBFGS(); autodiff=:forward)
    y = Optim.minimizer(res)
    return y
end;

#=
On the other hand, optimality conditions should be provided explicitly whenever possible, so as to avoid nesting autodiff calls.
=#

zero_gradient(x, y) = 2(y - x);

# We now have all the ingredients to construct our implicit function.

implicit = ImplicitFunction(dumb_identity, zero_gradient, gmres);

# ## Testing

x = rand(3, 2)

# Let's start by taking a look at the forward pass, which should be the identity function.

implicit(x)

# We now check whether the behavior of our `ImplicitFunction` wrapper is coherent with the theoretical derivatives.

Zygote.jacobian(implicit, x)[1]

# As expected, we recover the identity matrix as Jacobian. Strictly speaking, the Jacobian should be a 4D tensor, but it is flattened by Zygote into a 2D matrix.

# Note that implicit differentiation was necessary here, since our solver alone doesn't support autodiff with `Zygote.jl`.

try
    Zygote.jacobian(dumb_identity, x)[1]
catch e
    e
end

# The following tests are not included in the docs.  #src

@testset verbose = true "ChainRulesTestUtils" begin  #src
    test_frule(implicit, x; check_inferred=true)  #src
    test_rrule(implicit, x; check_inferred=true)  #src
end  #src
