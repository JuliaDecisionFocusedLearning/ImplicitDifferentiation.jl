# # Optimal transport

#=
In this example, we show how to differentiate through the solution of the entropy-regularized optimal transport problem.
=#

using Distances
using FiniteDifferences
using ImplicitDifferentiation
using Zygote

using LinearAlgebra #src
using Test  #src

#=
## Introduction

Here we give a brief introduction to optimal transport, see the [book by Gabriel Peyré and Marco Cuturi](https://optimaltransport.github.io/book/) for more details.

### Problem description

Suppose we have a distribution of mass ``a \in \Delta^{n}`` over points ``x_1, ..., x_{n} \in \mathbb{R}^d`` (where ``\Delta`` denotes the probability simplex).
We want to transport it to a distribution ``b \in \Delta^{m}`` over points ``y_1, ..., y_{m} \in \mathbb{R}^d``.
The unit moving cost from ``x`` to ``y`` is proportional to the squared Euclidean distance ``c(x, y) = \lVert x - y \rVert_2^2``.

A transportation plan can be described by a coupling ``p = \Pi(a, b)``, i.e. a probability distribution on the product space with the right marginals:
```math
\Pi(a, b) = \{p \in \Delta^{n \times m}: p \mathbf{1} = a, p^\top \mathbf{1} = b\}
```
Let ``C \in \mathbb{R}^{n \times m}`` be the moving cost matrix, with ``C_{ij} = c(x_i, y_j)``.
The basic optimization problem we want to solve is a linear program:
```math
\hat{p}(C) = \min_{p \in \Pi(a, b)} \sum_{i,j} p_{ij} C_{ij}
```
In order to make it smoother, we add an entropic regularization term:
 ```math
\hat{p}_{\varepsilon}(C) = \min_{p \in \Pi(a, b)} \sum_{i,j} \left(p_{ij} C_{ij} + \varepsilon p_{ij} \log \frac{p_{ij}}{a_i b_j} \right)
```

### Sinkhorn algorithm

To solve the regularized problem, we can use the Sinkhorn fixed point algorithm.
Let ``K \in \mathbb{R}^{n \times m}`` be the matrix defined by ``K_{ij} = \exp(-C_{ij} / \varepsilon)``.
Then the optimal coupling ``\hat{p}_{\varepsilon}(C)`` can be written as:
```math
\hat{p}_{\varepsilon}(C) = \mathrm{diag}(\hat{u}) ~ K ~ \mathrm{diag}(\hat{v}) \tag{1}
```
where ``\hat{u}`` and ``\hat{v}`` are the fixed points of the following Sinkhorn iteration:
```math
u^{t+1} = \frac{a}{Kv^t} \qquad \text{and} \qquad v^{t+1} = \frac{b}{K^\top u^t} \tag{S}
```

The implicit function theorem can be used to differentiate ``\hat{u}`` and ``\hat{v}`` with respect to ``C``, ``a`` and/or ``b``.
This can be combined with automatic differentiation of Equation (1) to find the Jacobian
```math
J = \frac{\partial ~ \mathrm{vec}(\hat{p}_{\varepsilon}(C))}{\partial ~ \mathrm{vec}(C)}
```
=#

d = 10
n = 3
m = 4

X = rand(d, n)
Y = rand(d, m)

a = fill(1 / n, n)
b = fill(1 / m, m)
C = pairwise(SqEuclidean(), X, Y, dims=2)

ε = 1.;

# ## Forward solver

# For technical reasons related to optimality checking, our Sinkhorn solver returns ``\hat{u}`` instead of ``\hat{p}_\varepsilon``.

function sinkhorn(C; a=a, b=b, ε=ε)
    K = exp.(.-C ./ ε)
    u = copy(a)
    v = copy(b)
    for t in 1:100
        u .= a ./ (K * v)
        v .= b ./ (K' * u)
    end
    return u
end

# ## Optimality conditions

# We simply used the fixed point equation $(\text{S})$.

function sinkhorn_fixed_point(C, u; a=a, b=b, ε=ε)
    K = exp.(.-C ./ ε)
    v = b ./ (K' * u)
    return u .- a ./ (K * v)
end

# We have all we need to build a differentiable Sinkhorn that doesn't require unrolling the fixed point iterations.

implicit = ImplicitFunction(sinkhorn, sinkhorn_fixed_point);

# ## Testing

u = sinkhorn(C)

# First, let us check that the forward pass works correctly and returns a fixed point.

maximum(abs, sinkhorn_fixed_point(C, u))

# Using the implicit function defined above, we can build an autodiff-compatible implementation of `transportation_plan` which does not require backpropagating through the Sinkhorn iterations:

function transportation_plan(C; a=a, b=b, ε=ε)
    K = exp.(.-C ./ ε)
    u = implicit(C)
    v = b ./ (K' * u)
    p_vec = vec(u .* K .* v')
    return p_vec
end;

# Let us compare its Jacobian with the one obtained using finite differences.

J = Zygote.jacobian(transportation_plan, C)[1]
J_ref = FiniteDifferences.jacobian(central_fdm(5, 1), transportation_plan, C)[1]
isapprox(J, J_ref, atol=1e-5)

# The following tests are not included in the docs.  #src

@testset verbose = true "FiniteDifferences" begin  #src
    @test isapprox(J, J_ref, atol=1e-2)  #src
end  #src
