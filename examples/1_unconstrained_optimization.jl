# # Unconstrained optimization

#=
In this example, we show how to differentiate through the solution of the following unconstrained optimization problem:
```math
\hat{y}(x) = \underset{y \in \mathbb{R}^m}{\mathrm{argmin}} ~ f(x, y)
```
The optimality conditions are given by gradient stationarity:
```math
F(x, \hat{y}(x)) = 0 \quad \text{with} \quad F(x,y) = \nabla_2 f(x, y) = 0
```
=#

using ChainRulesTestUtils  #src
using ForwardDiff
using ForwardDiffChainRules
using ImplicitDifferentiation
using LinearAlgebra  #src
using Optim
using Random
using Test  #src
using Zygote

Random.seed!(63);

# ## Implicit function wrapper

#=
To make verification easy, we minimize a quadratic objective
```math
f(x, y) = \lVert y - x \rVert^2
```
In this case, the optimization algorithm is very simple (the identity function does the job), but still we implement it using a black box solver from [Optim.jl](https://github.com/JuliaNLSolvers/Optim.jl) to show that it doesn't change the result.
=#

function dumb_identity(x::AbstractArray{Float64})
    f(y) = sum(abs2, y - x)
    y0 = zero(x)
    res = optimize(f, y0, LBFGS())
    y = Optim.minimizer(res)
    return y
end;

#=
On the other hand, optimality conditions should be provided explicitly whenever possible, so as to avoid nesting autodiff calls.
=#

zero_gradient(x, y) = 2(y - x);

# We now have all the ingredients to construct our implicit function.

implicit = ImplicitFunction(dumb_identity, zero_gradient);

# Time to test!

x = rand(3, 2)

# Let's start by taking a look at the forward pass, which should be the identity function.

implicit(x)

# ## Why bother?

# It is important to understand why implicit differentiation is necessary here. Indeed, our optimization solver alone doesn't support autodiff with ForwardDiff.jl (due to type constraints)

try
    ForwardDiff.jacobian(dumb_identity, x)
catch e
    e
end

# ... nor is it compatible with Zygote.jl (due to unsupported `try/catch` statements).

try
    Zygote.jacobian(dumb_identity, x)[1]
catch e
    e
end

# ## Autodiff with Zygote.jl

# If we use an autodiff package compatible with [ChainRules.jl](https://github.com/JuliaDiff/ChainRules.jl), such as [Zygote.jl](https://github.com/FluxML/Zygote.jl), implicit differentiation works out of the box.

Zygote.jacobian(implicit, x)[1]

# As expected, we recover the identity matrix as Jacobian. Strictly speaking, the Jacobian should be a 4D tensor, but it is flattened into a 2D matrix.

# ## Autodiff with ForwardDiff.jl

# If we want to use [ForwardDiff.jl](https://github.com/JuliaDiff/ForwardDiff.jl) instead, we run into a problem: custom chain rules are not directly translated into dual number dispatch. Luckily, [ForwardDiffChainRules.jl](https://github.com/ThummeTo/ForwardDiffChainRules.jl) provides us with a workaround. All we need to do is to apply the following macro:

@ForwardDiff_frule (f::typeof(implicit))(x::AbstractArray{<:ForwardDiff.Dual}; kwargs...)

# And then things work like a charm!

ForwardDiff.jacobian(implicit, x)

# ## Higher order differentiation

h = rand(size(x));

# Assuming we need second-order derivatives, nesting calls to Zygote.jl is generally a bad idea. We can, however, nest calls to ForwardDiff.jl.

D(x, h) = ForwardDiff.derivative(t -> implicit(x .+ t .* h), 0)
DD(x, h1, h2) = ForwardDiff.derivative(t -> D(x .+ t .* h2, h1), 0);

#-

try
    DD(x, h, h)  # fails
catch e
    e
end

# The only requirement is to switch to a linear solver that is compatible with dual numbers (which the default `gmres` from [Krylov.jl](https://github.com/JuliaSmoothOptimizers/Krylov.jl) is not).

linear_solver2(A, b) = (Matrix(A) \ b, (solved=true,))
implicit2 = ImplicitFunction(dumb_identity, zero_gradient, linear_solver2);
@ForwardDiff_frule (f::typeof(implicit2))(x::AbstractArray{<:ForwardDiff.Dual}; kwargs...)

D2(x, h) = ForwardDiff.derivative(t -> implicit2(x .+ t .* h), 0)
DD2(x, h1, h2) = ForwardDiff.derivative(t -> D2(x .+ t .* h2, h1), 0);

#-

DD2(x, h, h)

# The following tests are not included in the docs.  #src

@testset verbose = true "ForwardDiff.jl" begin  #src
    @test_throws MethodError ForwardDiff.jacobian(dumb_identity, x)  #src
    @test ForwardDiff.jacobian(implicit, x) == I  #src
    @test all(DD2(x, h, h) .≈ 0)  #src
end  #src

@testset verbose = true "Zygote.jl" begin  #src
    @test_throws Zygote.CompileError Zygote.jacobian(dumb_identity, x)[1]  #src
    @test Zygote.jacobian(implicit, x)[1] == I  #src
end  #src

@testset verbose = false "ChainRulesTestUtils.jl (forward)" begin  #src
    test_frule(implicit, x; check_inferred=true, rtol=1e-3)  #src
end  #src

@testset verbose = false "ChainRulesTestUtils.jl (reverse)" begin  #src
    test_rrule(implicit, x; check_inferred=true, rtol=1e-3)  #src
end  #src
