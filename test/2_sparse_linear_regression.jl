# # Sparse linear regression

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

using ComponentArrays
using Convex
using FiniteDifferences
using ImplicitDifferentiation
using MathOptInterface
using MathOptSetDistances
using Random
using SCS
using Zygote

using ChainRulesTestUtils  #src
using Test  #src

Random.seed!(63)

# ## Introduction

#=
We have a matrix of features $X \in \mathbb{R}^{n \times p}$ and a vector of targets $y \in \mathbb{R}^n$.

In a linear regression setting $y \approx X \beta$, one way to ensure sparsity of the parameter $\beta \in \mathbb{R}^p$ is to select it within the $\ell_1$ ball $\mathcal{B}_1$:
```math
\hat{\beta}(X, y) = \min_{\beta} ~ \lVert y - X \beta \rVert_2^2 \quad \text{s.t.} \quad \lVert \beta \rVert_1 \leq 1 \tag{QP}
```
We want to compute the derivatives of the optimal parameter wrt to the data: $\partial \hat{\beta} / \partial X$ and $\partial \hat{\beta} / \partial y$.

Possible application: sensitivity analysis of $\hat{\beta}(X, y)$.
=#

# ## Forward solver

# The function $\hat{\beta}$ is computed with a disciplined convex solver thanks to `Convex.jl`.

function lasso(X::AbstractMatrix, y::AbstractVector)
	n, p = size(X)
	β = Variable(p)
	objective = sumsquares(X * β - y)
	constraints = [norm(β, 1) <= 1.]
	problem = minimize(objective, constraints)
	solve!(problem, SCS.Optimizer; silent_solver=true)
	return Convex.evaluate(β)
end

# To comply with the requirements of `ImplicitDifferentiation.jl`, we need to provide the input arguments within a single array. We exploit `ComponentArrays.jl` for that purpose.

lasso(data::ComponentVector) = lasso(data.X, data.y)

# ## Optimality conditions

# We use `MathOptSetDistances.jl` to compute the projection onto the unit $\ell_1$ ball.

function proj_l1_ball(v::AbstractVector{R}) where {R<:Real}
	distance = MathOptSetDistances.DefaultDistance()
    cone = MathOptInterface.NormOneCone(length(v))
	ball = MathOptSetDistances.NormOneBall{R}(one(R), cone)
	return projection_on_set(distance, v, ball)
end

# Since this projection uses mutation internally, it is not compatible with `Zygote.jl`. Thus, we need to specify that it should be differentiated with `ForwardDiff.jl`.

function proj_grad_fixed_point(data, β)
	grad = 2 * data.X' * (data.X * β - data.y)
	return β - Zygote.forwarddiff(proj_l1_ball, β - grad)
end

# This is the last ingredient we needed to build a differentiable sparse linear regression.

implicit = ImplicitFunction(lasso, proj_grad_fixed_point);

# ## Testing

n, p = 5, 7;
X, y = rand(n, p), rand(n);
data = ComponentVector(X=X, y=y);

# As expected, the forward pass returns a sparse solution

round.(implicit(data); digits=4)

# Note that implicit differentiation is necessary here because the convex solver breaks autodiff.

try
    Zygote.jacobian(lasso, data)
catch e
    e
end

# Meanwhile, our implicit wrapper makes autodiff work seamlessly.

J = Zygote.jacobian(implicit, data)[1]

# The number of columns of the Jacobian is explained by the following formula:

prod(size(X)) + prod(size(y))

# We can validate the result using finite differences.

J_ref = FiniteDifferences.jacobian(central_fdm(5, 1), lasso, data)[1]
sum(abs, J - J_ref) / prod(size(J))

# The following tests are not included in the docs.  #src

@testset verbose = true "FiniteDifferences" begin  #src
    @test sum(abs, J - J_ref) / prod(size(J)) <= 1e-2  #src
end  #src
