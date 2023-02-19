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

# ## Implicit function wrapper

#=
To make verification easy, we minimize a quadratic objective
```math
f(x, y) = \lVert y - x \rVert^2
```
In this case, the optimization algorithm is very simple (the identity function does the job), but still we implement it using a black box solver from [Optim.jl](https://github.com/JuliaNLSolvers/Optim.jl) to show that it doesn't change the result.
=#

using Optim

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

using ImplicitDifferentiation

implicit = ImplicitFunction(dumb_identity, zero_gradient);

# Time to test!

using Random
Random.seed!(63)

x = rand(3, 2)

# Let's start by taking a look at the forward pass, which should be the identity function.

implicit(x)

# ## Autodiff with Zygote.jl

using Zygote

# If we use an autodiff package compatible with [ChainRules.jl](https://github.com/JuliaDiff/ChainRules.jl), such as [Zygote.jl](https://github.com/FluxML/Zygote.jl), differentiation works out of the box.

Zygote.jacobian(implicit, x)[1]

# As expected, we recover the identity matrix as Jacobian. Strictly speaking, the Jacobian should be a 4D tensor, but it is flattened into a 2D matrix.

# ## Autodiff with ForwardDiff.jl

using ForwardDiff

# If we want to use [ForwardDiff.jl](https://github.com/JuliaDiff/ForwardDiff.jl) instead, we run into a problem: custom chain rules are not directly translated into dual number dispatch. Luckily, [ForwardDiffChainRules.jl](https://github.com/ThummeTo/ForwardDiffChainRules.jl) provides us with a workaround. All we need to do is to apply the following macro:

using ForwardDiffChainRules

@ForwardDiff_frule (f::typeof(implicit))(x::AbstractArray{<:ForwardDiff.Dual}; kwargs...)

# And then things work like a charm!

ForwardDiff.jacobian(implicit, x)

# ## Why did we bother?

# It is important to understand that implicit differentiation was necessary here. Indeed our solver alone doesn't support autodiff with ForwardDiff.jl (due to type constraints)

try
    ForwardDiff.jacobian(dumb_identity, x)
catch e
    e
end

# ... nor was it compatible with Zygote.jl (due to unsupported `try/catch` statements).

try
    Zygote.jacobian(dumb_identity, x)[1]
catch e
    e
end

# The following tests are not included in the docs.  #src

using ChainRulesTestUtils  #src
using LinearAlgebra  #src
using Test  #src

@testset verbose = true "ForwardDiff.jl" begin  #src
    @test_throws MethodError ForwardDiff.jacobian(dumb_identity, x)  #src
    @test ForwardDiff.jacobian(implicit, x) == I  #src
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
