# # Constrained optimization

#=
In this example, we show how to differentiate through the solution of the following constrained optimization problem:
```math
\hat{y}(x) = \min_{y \in \mathcal{C}} f(x, y)
```
where ``\mathcal{C}`` is a closed convex set.
The optimal solution can be found as the fixed point of the projected gradient algorithm for any step size ``\eta``. This insight yields the following optimality conditions:
```math
F(x, \hat{y}(x)) = 0 \quad \text{with} \quad F(x,y) = \mathrm{proj}_{\mathcal{C}}(y - \eta \nabla_2 f(x, y)) - y
```
=#

using ChainRulesCore
using ImplicitDifferentiation
using Ipopt
using JuMP
using Krylov: gmres
using Zygote

using ChainRulesTestUtils  #src
using Test  #src

# ## Projecting onto the simplex

#=
We focus on minimizing ``f(x,y) = \lVert y - x \rVert_2^2``.
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

# ## Implicit function wrapper

# We now wrap a black box optimizer inside an `ImplicitFunction` to compare its implicit differentiation with the explicit procedure given above.

function forward(x)
    n = length(x)
    model = Model(optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0))
    @variable(model, y[1:n] >= 0)
    @constraint(model, sum(y) == 1)
    @objective(model, Min, sum((y .- x) .^ 2))
    optimize!(model)
    return value.(y)
end;

conditions(x, y) = simplex_projection(y - 0.1*2(y - x)) - y;

implicit = ImplicitFunction(; forward=forward, conditions=conditions, linear_solver=gmres);

# ## Testing

# Let us study the behavior of our implicit function.

x = rand(5)

# We can see that the forward pass computes the projection correctly, at least up to numerical precision.

hcat(simplex_projection(x), implicit(x))

# And the same goes for the Jacobian.

cat(
    Zygote.jacobian(simplex_projection, x)[1],
    Zygote.jacobian(implicit, x)[1],
    dims=3
)

# The following tests are not included in the docs.  #src

test_rrule(simplex_projection, x)  #src
@test implicit(x) ≈ simplex_projection(x) atol = 1e-5  #src
test_rrule(implicit, x; atol=1e-2)  #src
