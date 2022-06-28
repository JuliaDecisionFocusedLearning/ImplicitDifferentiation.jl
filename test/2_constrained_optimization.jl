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

using ImplicitDifferentiation
using Ipopt
using JuMP
using Zygote

using ChainRulesTestUtils  #src
using Test  #src

# ## Projecting onto the simplex

#=
We focus on minimizing ``f(x,y) = \lVert x - y \rVert_2^2``.
We also assume that ``\mathcal{C} = \Delta^n`` is the ``n``-dimensional probability simplex, because we know an exact procedure to compute the projection *and* its Jacobian.

Both of these procedures are outlined in [_From Softmax to Sparsemax: A Sparse Model of Attention and Multi-Label Classification_](https://arxiv.org/abs/1602.02068).
You can find an implementation in the function [`simplex_projection`](@ref).
Because this function involves a call to `sort`, standard AD backends cannot differentiate through it, which is why we also had to define a chain rule for it.
=#

# ## Implicit function wrapper

# We now wrap a black box optimizer inside an `ImplicitFunction` to compare its implicit differentiation with the explicit procedure given above.

function forward(x)
    n = length(x)
    optimizer = optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0)
    model = Model(optimizer)
    @variable(model, y[1:n] >= 0)
    @constraint(model, sum(y) == 1)
    @objective(model, Min, sum((y .- x) .^ 2))
    optimize!(model)
    return value.(y)
end;

conditions(x, y) = simplex_projection(y - 0.1 * 2(y - x)) - y;

implicit = ImplicitFunction(forward, conditions);

# ## Testing

# Let us study the behavior of our implicit function.

x = rand(5)

# We can see that the forward pass computes the projection correctly, at least up to numerical precision.

hcat(simplex_projection(x), implicit(x))

# And the same goes for the Jacobian.

cat(Zygote.jacobian(simplex_projection, x)[1], Zygote.jacobian(implicit, x)[1]; dims=3)

# The following tests are not included in the docs.  #src

@test implicit(x) ≈ simplex_projection(x) atol = 1e-5  #src

@testset verbose = true "ChainRules" begin  #src
    test_rrule(simplex_projection, x; check_inferred=true)  #src
    test_rrule(implicit, x; atol=1e-2, check_inferred=true)  #src
end  #src
