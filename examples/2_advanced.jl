# # Advanced use cases

#=
We dive into more advanced applications of implicit differentiation:
- constrained optimization problems
=#

using ForwardDiff
using ImplicitDifferentiation
using LinearAlgebra
using Optim
using Random
using Test  #src
using Zygote

Random.seed!(63);

# ## Constrained optimization

#=
First, we show how to differentiate through the solution of a constrained optimization problem:
```math
y(x) = \underset{y \in \mathbb{R}^m}{\mathrm{argmin}} ~ f(x, y) \quad \text{subject to} \quad g(x, y) \leq 0
```
The optimality conditions are a bit trickier than in the previous cases.
We can projection on the feasible set $\mathcal{C}(x) = \{y: g(x, y) \leq 0 \}$ and exploit the convergence of projected gradient descent with step size $\eta$:
```math
y = \mathrm{proj}_{\mathcal{C}(x)} (y - \eta \nabla_2 f(x, y))
```
=#

#=
To make verification easy, we minimize the following objective:
```math
f(x, y) = \lVert y \odot y - x \rVert^2
```
on the hypercube $\mathcal{C}(x) = [0, 1]^n$.
In this case, the optimization problem boils down to a thresholded componentwise square root function, but we implement it using a black box solver from [Optim.jl](https://github.com/JuliaNLSolvers/Optim.jl).
=#

function forward_cstr_optim(x)
    f(y) = sum(abs2, y .^ 2 - x)
    lower = zeros(size(x))
    upper = ones(size(x))
    y0 = ones(eltype(x), size(x)) ./ 2
    res = optimize(f, lower, upper, y0, Fminbox(GradientDescent()))
    y = Optim.minimizer(res)
    return y
end

#-

function proj_hypercube(p)
    return max.(0, min.(1, p))
end

function conditions_cstr_optim(x, y)
    ∇₂f = @. 4 * (y^2 - x) * y
    η = 0.1
    return y .- proj_hypercube(y .- η .* ∇₂f)
end

# We now have all the ingredients to construct our implicit function.

implicit_cstr_optim = ImplicitFunction(forward_cstr_optim, conditions_cstr_optim)

# And indeed, it behaves as it should when we call it:

x = rand(2) .+ [0, 1]

#=
The second component of $x$ is $> 1$, so its square root will be thresholded to one, and the corresponding derivative will be $0$.
=#

implicit_cstr_optim(x) .^ 2
@test implicit_cstr_optim(x) .^ 2 ≈ [x[1], 1]  #src

#-

J_thres = Diagonal([0.5 / sqrt(x[1]), 0])

# Forward mode autodiff

ForwardDiff.jacobian(implicit_cstr_optim, x)
@test ForwardDiff.jacobian(implicit_cstr_optim, x) ≈ J_thres  #src

#-

ForwardDiff.jacobian(forward_cstr_optim, x)

# Reverse mode autodiff

Zygote.jacobian(implicit_cstr_optim, x)[1]
@test Zygote.jacobian(implicit_cstr_optim, x)[1] ≈ J_thres  #src

#-

try
    Zygote.jacobian(forward_cstr_optim, x)[1]
catch e
    e
end
