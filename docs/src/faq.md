# Frequently Asked Questions

## Supported autodiff backends

| Backend                                                                | Forward mode | Reverse mode |
| ---------------------------------------------------------------------- | ------------ | ------------ |
| [ForwardDiff.jl](https://github.com/JuliaDiff/ForwardDiff.jl)          | yes          | -            |
| [ChainRules.jl](https://github.com/JuliaDiff/ChainRules.jl)-compatible | yes          | soon         |
| [Enzyme.jl](https://github.com/EnzymeAD/Enzyme.jl)                     | someday      | someday      |

## Writing conditions

We recommend that the conditions themselves do not involve calls to autodiff, even when they describe a gradient.
Otherwise, you will need to make sure that nested autodiff works well in your case.
For instance, if you're differentiating your implicit function in reverse mode with Zygote.jl, you may want to use [`Zygote.forwarddiff`](https://fluxml.ai/Zygote.jl/stable/utils/#Zygote.forwarddiff) to wrap the conditions and differentiate them with ForwardDiff.jl instead.

## Matrices and higher-order arrays

For simplicity, our examples only display functions that eat and spit out vectors.
However, arbitrary array shapes are supported, as long as the forward mapping _and_ conditions return similar arrays.
Beware however, sparse arrays will be densified in the differentiation process.

## Scalars

Functions that eat or spit out a single number are not supported.
The forward mapping _and_ conditions need arrays: for example, instead of returning `val` you should return `[val]` (a 1-element `Vector`).

## Multiple inputs / outputs

In this package, implicit functions can only take a single input array `x` and output a single output array `y` (plus the byproduct `z`).
But sometimes, your forward mapping or conditions may require multiple input arrays, say `a` and `b`:

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

## Using byproducts

Why would the forward mapping return a byproduct `z` in addition to `y`?
It is mainly useful when the solution procedure creates objects such as Jacobians, which we want to reuse when computing or differentiating the conditions.
In that case, you may want to write the differentiation rules yourself for the conditions.
A more advanced application is given by [DifferentiableFrankWolfe.jl](https://github.com/gdalle/DifferentiableFrankWolfe.jl).

Keep in mind that derivatives of `z` will not be computed: the byproduct is considered constant during differentiation (unlike the case of multiple outputs outlined above).

## Performance tips

If you work with small arrays (say, less than 100 elements), consider using [StaticArrays.jl](https://github.com/JuliaArrays/StaticArrays.jl) if you seek increased performance.

## Modeling tips

To express constrained optimization problems as implicit functions, you might need differentiable projections or proximal operators to write the optimality conditions.
See [_Efficient and modular implicit differentiation_](https://arxiv.org/abs/2105.15183) for precise formulations.

In case these operators are too complicated to code them yourself, here are a few places you can look:

- [MathOptSetDistances.jl](https://github.com/matbesancon/MathOptSetDistances.jl)
- [ProximalOperators.jl](https://github.com/JuliaFirstOrder/ProximalOperators.jl)

An alternative is differentiating through the KKT conditions, which is exactly what [DiffOpt.jl](https://github.com/jump-dev/DiffOpt.jl) does for JuMP models.
