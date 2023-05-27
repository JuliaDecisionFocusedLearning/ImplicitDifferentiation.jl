# Frequently Asked Questions

## Supported autodiff backends

- Forward mode: [ForwardDiff.jl](https://github.com/JuliaDiff/ForwardDiff.jl)
- Reverse mode: all the packages compatible with [ChainRules.jl](https://github.com/JuliaDiff/ChainRules.jl)

In the future, we would like to add [Enzyme.jl](https://github.com/EnzymeAD/Enzyme.jl) support.

## Higher-dimensional arrays

For simplicity, our examples only display functions that eat and spit out vectors.
However, arbitrary array shapes are supported, as long as the forward _and_ conditions callables return similar arrays.
Beware however, sparse arrays will be densified in the differentiation process.

## Scalar input / output

Functions that eat or spit out a single number are not supported.
The forward _and_ conditions callables need arrays: for example, instead of returning `value` you should return `[value]` (a 1-element `Vector`). 
Consider using an `SVector` from [StaticArrays.jl](https://github.com/JuliaArrays/StaticArrays.jl) if you seek increased performance.

## Multiple inputs / outputs

In this package, implicit functions can only take a single input array `x` and output a single output array `y` (plus the byproduct `z`).
But sometimes, your forward pass or conditions may require multiple input arrays, say `a` and `b`:

```julia
function f(a, b)
    # do stuff
    return y, z
end
```

In that case, you should gather the inputs inside a single `ComponentVector` from [ComponentArrays.jl](https://github.com/jonniedie/ComponentArrays.jl) and define a new method:

```julia
f(x::ComponentVector) = f(x.a, x.b)
```

The same trick works for multiple outputs.

## Byproducts

At first glance, it is not obvious why we impose that the `forward` callable returns a byproduct `z` in addition to `y`.
It is mainly useful when the solution procedure creates objects such as Jacobians, which we want to reuse when computing or differentiating the `conditions`.
We will provide simple examples soon.
In the meantime, an advanced application is given by [DifferentiableFrankWolfe.jl](https://github.com/gdalle/DifferentiableFrankWolfe.jl).

## Modeling constrained optimization problems

To express constrained optimization problems as implicit functions, you might need differentiable projections or proximal operators to write the optimality conditions.
See [_Efficient and modular implicit differentiation_](https://arxiv.org/abs/2105.15183) for precise formulations.

In case these operators are too complicated to code them yourself, here are a few places you can look:

- [MathOptSetDistances.jl](https://github.com/matbesancon/MathOptSetDistances.jl)
- [ProximalOperators.jl](https://github.com/JuliaFirstOrder/ProximalOperators.jl)

An alternative is differentiating through the KKT conditions, which is exactly what [DiffOpt.jl](https://github.com/jump-dev/DiffOpt.jl) does for JuMP models.
