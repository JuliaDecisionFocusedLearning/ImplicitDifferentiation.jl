# # Optimal transport

#=
In this example, we show how to differentiate through the solution of the entropy-regularized optimal transport problem.
=#

using ImplicitDifferentiation
using OptimalTransport
using Distances
using Krylov: gmres
using Zygote
using FiniteDiff
using Test, LinearAlgebra #src

#=
## Introduction

Here we give a brief introduction to optimal transport, see the [book by Gabriel Peyré and Marco Cuturi](https://optimaltransport.github.io/book/) for more details.

### Problem description

Suppose we have a distribution of mass ``\mu \in \Delta^{n}`` over points ``x_1, ..., x_{n} \in \mathbb{R}^d`` (where ``\Delta`` denotes the probability simplex).
We want to transport it to a distribution ``\nu \in \Delta^{m}`` over points ``y_1, ..., y_{m} \in \mathbb{R}^d``.
The unit moving cost from ``x`` to ``y`` is proportional to the squared Euclidean distance ``c(x, y) = \lVert x - y \rVert_2^2``.

A transportation plan can be described by a coupling ``p = \Pi(\mu, \nu)``, i.e. a probability distribution on the product space with the right marginals:
```math
\Pi(\mu, \nu) = \{p \in \Delta^{n \times m}: p \mathbf{1} = \mu, p^\top \mathbf{1} = \nu\}
```
Let ``C \in \mathbb{R}^{n \times m}`` be the moving cost matrix, with ``C_{ij} = c(x_i, y_j)``.
The basic optimization problem we want to solve is a linear program:
```math
\hat{p}(C) = \min_{p \in \Pi(\mu, \nu)} \sum_{i,j} p_{ij} C_{ij}
```
In order to make it smoother, we add an entropic regularization term:
 ```math
\hat{p}_{\varepsilon}(C) = \min_{p \in \Pi(\mu, \nu)} \sum_{i,j} \left(p_{ij} C_{ij} + \varepsilon p_{ij} \log \frac{p_{ij}}{\mu_i \nu_j} \right)
```

### Sinkhorn algorithm

To solve the regularized problem, we can use the Sinkhorn fixed point algorithm.
Let ``K \in \mathbb{R}^{n \times m}`` be the matrix defined by ``K_{ij} = \exp(-C_{ij} / \varepsilon)``.
Then the optimal coupling ``\hat{p}_{\varepsilon}(C)`` can be written as:
```math
\hat{p}_{\varepsilon}(C) = \mathrm{diag}(\hat{u}) ~ K ~ \mathrm{diag}(\hat{v}) \tag{1}
```
where ``\hat{u}`` and ``\hat{v}`` are the fixed points of the following Sinkhorn fixed point iteration:
```math
u^{t+1} = \frac{\mu}{Kv^t} \qquad \text{and} \qquad v^{t+1} = \frac{\nu}{K^\top u^t}
```

The implicit function theorem can be used to differentiate ``\hat{u}`` and ``\hat{v}`` with respect to ``C``, ``\mu`` and/or ``\nu``.
This can be combined with automatic differentiation of Equation (1) to find the Jacobian
```math
J = \frac{\partial ~ \mathrm{vec}(\hat{p}_{\varepsilon}(C))}{\partial ~ \mathrm{vec}(C)}
```
=#

# ## Implicit function wrapper

# For now, `ImplicitFunction` objects do not take multiple arguments, so we use non-constant global variables instead (even though we shouldn't)

d = 10
n = 3
m = 4

X = rand(d, n)
Y = rand(d, m)

μ = fill(1 / n, n)
ν = fill(1 / m, m)
C_vec = vec(pairwise(SqEuclidean(), X, Y, dims=2))

ε = 1.0;

#=
We now embed the Sinkhorn algorithm and optimality conditions inside an `ImplicitFunction` struct.
For technical reasons related to optimality checking, our forward procedure returns ``\hat{u}`` instead of ``\hat{p}_\varepsilon``.
=#

function forward(C_vec)
    C = reshape(C_vec, n, m)
    solver = OptimalTransport.build_solver(μ, ν, C, ε, SinkhornGibbs())
    OptimalTransport.solve!(solver)
    û = solver.cache.u
    return û, nothing
end

function conditions(C_vec, û, useful_info=nothing)
    C = reshape(C_vec, n, m)
    K = exp.(.-C ./ ε)
    v̂ = ν ./ (K' * û)
    return û .- μ ./ (K * v̂)
end

implicit = ImplicitFunction(; forward=forward, conditions=conditions, linear_solver=gmres);

# ## Testing

# First, let us check that the forward pass works correctly

û, _ = forward(C_vec)

maximum(abs, conditions(C_vec, û))

# Using the implicit function defined above, we can build an autodiff-compatible implementation of `transportation_plan` which does not require backpropagating through the Sinkhorn iterations:

function transportation_plan(C_vec)
    C = reshape(C_vec, n, m)
    K = exp.(.-C ./ ε)
    û = implicit(C_vec)
    v̂ = ν ./ (K' * û)
    p̂_vec = vec(û .* K .* v̂')
    return p̂_vec
end;

# Let us compare with the result obtained using finite differences:

J_autodiff = Zygote.jacobian(transportation_plan, C_vec)[1]
J_finitediff = FiniteDiff.finite_difference_jacobian(transportation_plan, C_vec)
maximum(abs, J_autodiff - J_finitediff)

# The following tests are not included in the docs.  #src

@test maximum(abs, J_autodiff - J_finitediff) < 1e-7  #src
