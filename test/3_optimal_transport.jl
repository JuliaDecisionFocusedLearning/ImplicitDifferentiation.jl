# # Optimal transport

#=
In this example, we show how to differentiate through the solution of the entropy-regularized optimal transport problem:
```math
\hat{p}(M) = \min_{p \in U(r, c)} \sum_{i=1}^d_1 \sum_{j=1}^d_2 m(x_i, y_j) p_{ij} + \epsilon \sum_{i=1}^d_1 \sum_{j=1}^d_2 log \Big( \frac{p_{ij}}{r_i c_j} \Big)
```
where ``U(r, c) = \{ p \in [0,1]^{d_1 \times d_2} \mid p \mathbf{1} = r, p^\top \mathbf{1} = c\}`` is the set of doubly stochastic matrices with marginals ``r`` and ``c``.

The optimal solution can be found using the Sinkhorn fixed point algorithm. Let ``M`` be the matrix of size ``d_1 \times d_2`` such that ``M_{ij} = e^{-C_{ij} / \epsilon}`` and ``C_{ij} = m(x_i, y_j)``. The optimal solution ``p`` can be written as:
```math
p = \text{diag}(u) M \text{diag}(v)
```
where ``u`` and ``v`` are the fixed points of the following Sinkhorn fixed point iterations:
```math
u = r ./ (M \times v))
```
```math
v = c ./ (M^\top \times u))
```
The implicit function theorem can be used to differentiate ``u`` and ``v`` wrt ``C``, ``r`` and/or ``c``. This can be combined with automatic differentiation to find the Jacobian ``d(vec(p))/d(vec(C))`` for instance using the explicit function that maps ``u``, ``v`` and ``C`` to ``p``, where ``vec(p)`` is the vectorized optimal transport plan ``p`` and ``vec(C)`` is the vectorized distance matrix.
=#

using ChainRulesCore
using ImplicitDifferentiation
using OptimalTransport
using Distances
using Krylov: gmres
using Zygote
using ChainRulesTestUtils

# ## Implicit function wrapper

# We now call the Sinkhorn algorithm and define the optimality conditions using the `ImplicitFunction` struct.

n, m = 3, 3
X = rand(10, n)
Y = rand(10, m)
vC = vec(pairwise(SqEuclidean(), X, Y))
r = fill(1 / n, n)
c = fill(1 / m, m)
ϵ = 1.0
function forward(vC)
    C = reshape(vC, n, m)
    solver = OptimalTransport.build_solver(r, c, C, ϵ, SinkhornGibbs())
    OptimalTransport.solve!(solver)
    u = solver.cache.u
    return u
end
function conditions(vC, u)
    C = reshape(vC, n, m)
    M = exp.(-C ./ ϵ)
    v = c ./ (M' * u)
    return u .- r ./ (M * v)
end

implicit = ImplicitFunction(; forward=forward, conditions=conditions, linear_solver=gmres);

uv = forward(vC)
conditions(vC, uv)

# ## Testing

# Let us find the Jacobian `d(vec(P)) / d(vec(C))` using the implicit function defined above and the formula for the optimal transport plan given ``u``, ``v`` and ``C``.

function plan(vC)
    u = implicit(vC)
    C = reshape(vC, n, m)
    M = exp.(.-C ./ ϵ)
    v = c ./ (M' * u)
    vec(u .* M .* v')
end
Zygote.jacobian(plan, vC)[1]

# The following tests are not included in the docs.  #src

test_rrule(implicit, vC)  #src
