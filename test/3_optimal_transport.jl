# # Optimal transport

#=
In this example, we show how to differentiate through the solution of the entropy-regularized optimal transport problem.
=#

using Distances
using FiniteDifferences
using ImplicitDifferentiation
using LinearAlgebra
using Random
using Zygote

Random.seed!(63);

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
\hat{p}(C) = \underset{p \in \Pi(a, b)}{\mathrm{argmin}} ~ \sum_{i,j} p_{ij} C_{ij}
```
In order to make it smoother, we add an entropic regularization term:
 ```math
\hat{p}_{\varepsilon}(C) = \underset{p \in \Pi(a, b)}{\mathrm{argmin}} ~ \sum_{i,j} \left(p_{ij} C_{ij} + \varepsilon p_{ij} \log \frac{p_{ij}}{a_i b_j} \right)
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
C = pairwise(SqEuclidean(), X, Y; dims=2)

ε = 1.0;
T = 100;

# ## Forward solver

# For technical reasons related to optimality checking, our Sinkhorn solver returns ``\hat{u}`` instead of ``\hat{p}_\varepsilon``.

function sinkhorn(C; a, b, ε, T)
    K = exp.(.-C ./ ε)
    u = copy(a)
    v = copy(b)
    for t in 1:T
        u = a ./ (K * v)
        v = b ./ (K' * u)
    end
    return u
end

function sinkhorn_efficient(C; a, b, ε, T)
    K = exp.(.-C ./ ε)
    u = copy(a)
    v = copy(b)
    for t in 1:T
        mul!(u, K, v)
        u .= a ./ u
        mul!(v, K', u)
        v .= b ./ v
    end
    return u
end

# ## Optimality conditions

# We simply used the fixed point equation $(\text{S})$.

function sinkhorn_fixed_point(C, u; a, b, ε, T=nothing)
    K = exp.(.-C ./ ε)
    v = b ./ (K' * u)
    return u .- a ./ (K * v)
end

# We have all we need to build a differentiable Sinkhorn that doesn't require unrolling the fixed point iterations.

implicit = ImplicitFunction(sinkhorn_efficient, sinkhorn_fixed_point);

# ## Testing

u1 = sinkhorn(C; a=a, b=b, ε=ε, T=T)
u2 = implicit(C; a=a, b=b, ε=ε, T=T)
u1 == u2

# First, let us check that the forward pass works correctly and returns a fixed point.

all(iszero, sinkhorn_fixed_point(C, u1; a=a, b=b, ε=ε, T=T))

# Using the implicit function defined above, we can build an autodiff-compatible Sinkhorn which does not require backpropagating through the fixed point iterations:

function transportation_plan_slow(C; a, b, ε, T)
    K = exp.(.-C ./ ε)
    u = sinkhorn(C; a=a, b=b, ε=ε, T=T)
    v = b ./ (K' * u)
    p = u .* K .* v'
    return p
end;

function transportation_plan_fast(C; a, b, ε, T)
    K = exp.(.-C ./ ε)
    u = implicit(C; a=a, b=b, ε=ε, T=T)
    v = b ./ (K' * u)
    p = u .* K .* v'
    return p
end;

# What does the transportation plan look like?

p1 = transportation_plan_slow(C; a=a, b=b, ε=ε, T=T)
p2 = transportation_plan_fast(C; a=a, b=b, ε=ε, T=T)
p1 == p2

# Let us compare its Jacobian with the one obtained using finite differences.

J1 = Zygote.jacobian(C -> transportation_plan_slow(C; a=a, b=b, ε=ε, T=T), C)[1]
J2 = Zygote.jacobian(C -> transportation_plan_fast(C; a=a, b=b, ε=ε, T=T), C)[1]
J_ref = FiniteDifferences.jacobian(
    central_fdm(5, 1), C -> transportation_plan_slow(C; a=a, b=b, ε=ε, T=T), C
)[1]

sum(abs, J2 - J_ref) / prod(size(J_ref))

# The following tests are not included in the docs.  #src

using Test  #src

@testset verbose = true "FiniteDifferences.jl" begin  #src
    @test u1 == u2  #src
    @test all(iszero, sinkhorn_fixed_point(C, u1; a=a, b=b, ε=ε, T=T))  #src
    @test p1 == p2  #src
    @test sum(abs, J1 - J_ref) / prod(size(J_ref)) < 1e-5  #src
    @test sum(abs, J2 - J_ref) / prod(size(J_ref)) < 1e-5  #src
end  #src
