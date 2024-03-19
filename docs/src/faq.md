# Frequently Asked Questions

## Supported autodiff backends

To differentiate an `ImplicitFunction`, the following backends are supported.

| Backend                                                                | Forward mode | Reverse mode |
| ---------------------------------------------------------------------- | ------------ | ------------ |
| [ForwardDiff.jl](https://github.com/JuliaDiff/ForwardDiff.jl)          | yes          | -            |
| [ChainRules.jl](https://github.com/JuliaDiff/ChainRules.jl)-compatible | soon          | yes         |
| [Enzyme.jl](https://github.com/EnzymeAD/Enzyme.jl)                     | someday      | someday      |

By default, the conditions are differentiated with the same backend as the `ImplicitFunction` that contains them.
However, this can be switched to any backend compatible with [AbstractDifferentiation.jl](https://github.com/JuliaDiff/AbstractDifferentiation.jl) (i.e. a subtype of `AD.AbstractBackend`).
You can specify it with the `conditions_backend` keyword argument when constructing an `ImplicitFunction`.

!!! warning "Warning"
    At the moment, `conditions_backend` can only be `nothing` or `AD.ForwardDiffBackend()`. We are investigating why the other backends fail.

## Input and output types

### Arrays

Functions that eat or spit out arbitrary arrays are supported, as long as the forward mapping _and_ conditions return arrays of the same size.

If you deal with small arrays (say, less than 100 elements), consider using [StaticArrays.jl](https://github.com/JuliaArrays/StaticArrays.jl) for increased performance.

### Scalars

Functions that eat or spit out a single number are not supported.
The forward mapping _and_ conditions need arrays: instead of returning `val` you should return `[val]` (a 1-element `Vector`).
Or better yet, wrap it in a static vector: `SVector(val)`.

### Sparse arrays

!!! danger "Danger"
    Sparse arrays are not officially supported and might give incorrect values or `NaN`s!

With ForwardDiff.jl, differentiation of sparse arrays will always give wrong results due to [sparsity pattern cancellation](https://github.com/JuliaDiff/ForwardDiff.jl/issues/658).
With Zygote.jl it appears to work, but this functionality is considered experimental and might evolve.

## Number of inputs and outputs

Most of the documentation is written for the simple case where the forward mapping is `x -> y`, i.e. one input and one output.
What can you do to handle multiple inputs or outputs?
Well, it depends whether you want their derivatives or not.

|                      | Derivatives needed                      | Derivatives not needed                  |
| -------------------- | --------------------------------------- | --------------------------------------- |
| **Multiple inputs**  | Make `x` a `ComponentVector`            | Supply `args` and `kwargs` to `forward` |
| **Multiple outputs** | Make `y` and `c` two `ComponentVector`s | Let `forward` return a byproduct        |

We now detail each of these options.

### Multiple inputs or outputs | Derivatives needed

Say your forward mapping takes multiple input arrays and returns multiple output arrays, such that you want derivatives for all of them.

The trick is to leverage [ComponentArrays.jl](https://github.com/jonniedie/ComponentArrays.jl) to wrap all the inputs inside a single a `ComponentVector`, and do the same for all the outputs.
See the examples for a demonstration.

!!! warning "Warning"
    You may run into issues trying to differentiate through the `ComponentVector` constructor.
    For instance, Zygote.jl will throw `ERROR: Mutating arrays is not supported`.
    Check out [this issue](https://github.com/gdalle/ImplicitDifferentiation.jl/issues/67) for a dirty workaround involving custom chain rules for the constructor.

### Multiple inputs | Derivatives not needed

If your forward mapping (or conditions) takes multiple inputs but you don't care about derivatives, then you can add further positional and keyword arguments beyond `x`.
It is important to make sure that the forward mapping and conditions accept the same set of arguments, even if each of these functions only uses a subset of them.

```julia
forward(x, arg1, arg2; kwarg1, kwarg2) = y
conditions(x, arg1, arg2; kwarg1, kwarg2) = c
```

All of the positional and keyword arguments apart from `x` will get zero tangents during differentiation of the implicit function.

## Multiple outputs | Derivatives not needed

The last and most tricky situation is when your forward mapping returns multiple outputs, but you only care about some of their derivatives.
Then, you need to group the objects you don't want to differentiate into a "byproduct" `z`, returned alongside the actual output `y`.
This way, derivatives of `z` will not be computed: the byproduct is considered constant during differentiation.

The signatures of your functions will need to be be slightly different from the previous cases:

```julia
forward(x, arg1, arg2; kwarg1, kwarg2) = (y, z)
conditions(x, y, z, arg1, arg2; kwarg1, kwarg2) =  c
```

See the examples for a demonstration.

This is mainly useful when the solution procedure creates objects such as Jacobians, which we want to reuse when computing or differentiating the conditions.
In that case, you may want to write the conditions differentiation rules yourself.
A more advanced application is given by [DifferentiableFrankWolfe.jl](https://github.com/gdalle/DifferentiableFrankWolfe.jl).

## Modeling tips

### Writing conditions

We recommend that the conditions themselves do not involve calls to autodiff, even when they describe a gradient.
Otherwise, you will need to make sure that nested autodiff works well in your case.
For instance, if you're differentiating your implicit function (and your conditions) in reverse mode with Zygote.jl, you may want to use ForwardDiff.jl mode to compute gradients inside the conditions.

### Dealing with constraints

To express constrained optimization problems as implicit functions, you might need differentiable projections or proximal operators to write the optimality conditions.
See [_Efficient and modular implicit differentiation_](https://arxiv.org/abs/2105.15183) for precise formulations.

In case these operators are too complicated to code them yourself, here are a few places you can look:

- [MathOptSetDistances.jl](https://github.com/matbesancon/MathOptSetDistances.jl)
- [ProximalOperators.jl](https://github.com/JuliaFirstOrder/ProximalOperators.jl)

An alternative is differentiating through the KKT conditions, which is exactly what [DiffOpt.jl](https://github.com/jump-dev/DiffOpt.jl) does for JuMP models.

### Memoization

In some cases, performance might be increased by using memoization to prevent redundant calls to `forward`.
For instance, this is relevant when calculating large Jacobians with forward differentiation, where the computation happens in [chunks](https://juliadiff.org/ForwardDiff.jl/stable/user/advanced/#Configuring-Chunk-Size).
Packages such as [Memoize.jl](https://github.com/JuliaCollections/Memoize.jl) and [Memoization.jl](https://github.com/marius311/Memoization.jl) are useful for defining a memoized version of `forward`:

```julia
using Memoize
@memoize Dict forward(x, args...; kwargs...) = y
```
