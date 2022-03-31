# # Constrained optimization

#=
In this example, we show how to differentiate through the solution of the following constrained optimization problem:
```math
\hat{y}(x) = \min_{y \in \mathcal{C}} f(x, y)
```
where ``\mathcal{C}`` is a closed convex set.
The optimal solution can be found as the fixed point of the projected gradient algorithm for any step size ``\eta``. This insight yields the following optimality conditions:
```math
F(x, \hat{y}(x)) = 0 \quad \text{with} \quad F(x,y) = \mathrm{proj}_{\mathcal{C}}(y - \eta \nabla_1 f(x, y)) - y
```
=#

using ChainRulesCore
using GalacticOptim
using ImplicitDifferentiation
using Ipopt
using IterativeSolvers
using JuMP
using LinearAlgebra
using Optim
using Statistics
using Zygote

using ChainRulesTestUtils  #src
using Test  #src

# ## Projecting onto the simplex

#=
We focus on minimizing ``f(x,y) = \lVert x - y \rVert_2^2``.
We also assume that ``\mathcal{C} = \Delta^n`` is the ``n``-dimensional probability simplex, because we know exact procedures to compute the projection *and* its Jacobian.
See <https://arxiv.org/abs/1602.02068> for details.
=#

function simplex_projection_and_support(z::AbstractVector{<:Real})
    d = length(z)
    z_sorted = sort(z; rev=true)
    z_sorted_cumsum = cumsum(z_sorted)
    k = maximum(j for j in 1:d if (1 + j * z_sorted[j]) > z_sorted_cumsum[j])
    τ = (z_sorted_cumsum[k] - 1) / k
    p = max.(z .- τ, 0)
    s = [Int(p[i] > eps()) for i in 1:d]
    return p, s
end;

function simplex_projection(z::AbstractVector{<:Real})
    p, _ = simplex_projection_and_support(z)
    return p
end;

# Note that defining a custom chain rule for the projection is indeed necessary, since it contains a call to `sort` that Zygote cannot differentiate through.

function ChainRulesCore.rrule(::typeof(simplex_projection), z::AbstractVector{<:Real})
    p, s = simplex_projection_and_support(z)
    S = sum(s)
    function simplex_projection_pullback(dp)
        vjp = s .* (dp .- (dp's) / S)
        return (NoTangent(), vjp)
    end
    return p, simplex_projection_pullback
end;

test_rrule(simplex_projection, rand(100))  #src

# ## Implicit function wrapper

# We now wrap a black box optimizer inside an `ImplicitFunction` to compare its implicit differentiation with the explicit procedure given above.

function forward(x)
    n = length(x)
    model = Model(optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0))
    @variable(model, y[1:n] >= 0)
    @constraint(model, sum(y) == 1)
    @objective(model, Min, sum((x .- y) .^ 2))
    optimize!(model)
    return value.(y)
end;

conditions(x, y) = simplex_projection(y - 0.1(x - y)) - y;

implicit = ImplicitFunction(; forward=forward, conditions=conditions, linear_solver=gmres);

# ## Testing

# Let us compare the behavior of our implicit function on two different vectors.

x_pass, x_fail = ones(5), rand(5)

# We can see that the forward pass correctly computes the projection, at least up to numerical precision.

hcat(simplex_projection(x_pass), implicit(x_pass))

#

hcat(simplex_projection(x_fail), implicit(x_fail))

# However, the Jacobian behaves differently, and it is incorrect for `x_fail`.

cat(
    Zygote.jacobian(simplex_projection, x_pass)[1],
    Zygote.jacobian(implicit, x_pass)[1],
    dims=3
)

#

cat(
    Zygote.jacobian(simplex_projection, x_fail)[1],
    Zygote.jacobian(implicit, x_fail)[1],
    dims=3
)

#=
So what happened?
Well, when the Jacobian ``\partial_2 F(x, \hat{y}(x))`` is not invertible, the implicit function theorem no longer holds.
Unfortunately, this happens as soon as the projection is sparse, since some coordinates of ``x`` won't play any role in the value of ``\hat{y}(x)``.

Hopefully, in this case, the invalid result obtained through implicit differentiation can still be used as a heuristic.
=#

# The following tests are not included in the docs.  #src

@testset verbose = true "x_pass vs. x_fail" begin  #src
    @test implicit(x_pass) ≈ simplex_projection(x_pass) atol = 1e-5  #src
    @test implicit(x_fail) ≈ simplex_projection(x_fail) atol = 1e-5  #src
    @test Zygote.jacobian(implicit, x_pass)[1] ≈  #src
        Zygote.jacobian(simplex_projection, x_pass)[1]  #src
    @test !(  #src
        Zygote.jacobian(implicit, x_fail)[1] ≈  #src
        Zygote.jacobian(simplex_projection, x_fail)[1]  #src
    )  #src
end  #src

test_rrule(implicit, x_pass; atol=1e-3)  #src
