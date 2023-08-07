# Frequently Asked Questions

## Supported autodiff backends

To differentiate an `ImplicitFunction`, the following backends are supported.

| Backend                                                                | Forward mode | Reverse mode |
| ---------------------------------------------------------------------- | ------------ | ------------ |
| [ForwardDiff.jl](https://github.com/JuliaDiff/ForwardDiff.jl)          | yes          | -            |
| [ChainRules.jl](https://github.com/JuliaDiff/ChainRules.jl)-compatible | yes          | soon         |
| [Enzyme.jl](https://github.com/EnzymeAD/Enzyme.jl)                     | someday      | someday      |

By default, the conditions are differentiated with the same backend as the `ImplicitFunction` that contains them.
However, this can be switched to any backend compatible with [AbstractDifferentiation.jl](https://github.com/JuliaDiff/AbstractDifferentiation.jl) (i.e. a subtype of `AD.AbstractBackend`).
You can specify it with the `conditions_backend` keyword argument when constructing an `ImplicitFunction`.

!!! warning "Warning"
    At the moment, `conditions_backend` can only be `nothing` or `AD.ForwardDiffBackend()`. We are investigating why the other backends fail.

## Input and output types

### Arrays

Functions that eat or spit out arbitrary arrays are supported, as long as the forward mapping _and_ conditions return arrays of the same size.
Beware however, sparse arrays will be densified in the differentiation process.

If the output is a small array (say, less than 100 elements), consider using [StaticArrays.jl](https://github.com/JuliaArrays/StaticArrays.jl) for increased performance.

### Scalars

Functions that eat or spit out a single number are not supported.
The forward mapping _and_ conditions need arrays: instead of returning `val` you should return `[val]` (a 1-element `Vector`).
Or better yet, wrap it in a static vector: `SVector(val)`.

## Number of inputs and outputs

Most of the documentation is written for the simple case where the forward mapping is `x -> y`, i.e. one input and one output.
What can you do to handle multiple inputs or outputs?
Well, it depends whether you want their derivatives or not.

|                      | Derivatives needed                      | Derivatives not needed                  |
| -------------------- | --------------------------------------- | --------------------------------------- |
| **Multiple inputs**  | Make `x` a `ComponentVector`            | Supply `args` and `kwargs` to `forward` |
| **Multiple outputs** | Make `y` and `c` two `ComponentVector`s | Let `foward` return a byproduct         |

We now detail each of these options.

### Multiple inputs or outputs | Derivatives needed

Say your forward mapping requires multiple input arrays `(x1, x2, x3)` and returns multiple output arrays `(y4, y5)`.
And say that you want derivatives for all of them.

```julia
function forward_aux(x1, x2, x3)
    # do stuff
    return y4, y5
end

function conditions_aux(x1, x2, x3, y4, y5)
    # do stuff
    return c4, c5
end
```

The trick is to leverage [ComponentArrays.jl](https://github.com/jonniedie/ComponentArrays.jl) to wrap all the inputs inside a single a `ComponentVector`, and do the same for all the outputs.

```julia
function forward(x::ComponentVector)
    y4, y5 = forward_aux(x.x1, x.x2, x.x3)
    y = ComponentVector(; y4=y4, y5=y5)
    return y
end

function conditions(x::ComponentVector, y::ComponentVector)
    c4, c5 = conditions_aux(x.x1, x.x2, x.x3, y.y4, y.y5)
    c = ComponentVector(; c4=c4, c5=c5)
    return c
end
```

Then if you construct

```julia
implicit = ImplicitFunction(conditions, forward)
```

you can use it as follows:

```julia
x = ComponentVector(; x1=x1, x2=x2, x3=x3)
implicit(x)
```

!!! warning "Warning"
    You may run into issues trying to differentiate through the `ComponentVector` constructor, with an error message like `ERROR: Mutating arrays is not supported` for Zygote.jl.
    Check out [this issue](https://github.com/gdalle/ImplicitDifferentiation.jl/issues/67) for a dirty workaround.

### Multiple inputs | Derivatives not needed

If you have multiple inputs but you don't care about derivatives, then you can add further positional and keyword arguments beyond `x`.
It is important to make sure that the forward mapping and conditions accept the same set of arguments, even if each of these functions only uses a subset.

```julia
forward(x, arg1, arg2; kwarg1, kwarg2) =  # do stuff, return y
conditions(x, arg1, arg2; kwarg1, kwarg2) =  # do stuff, return y
```

All of the positional and keyword arguments apart from `x` will get zero tangents during differentiation of the implicit function.

## Multiple outputs | Derivatives not needed

The same goes for the conditions, as in their first two (possibly three) positional arguments must be `x` and the output of the forward mapping  `y` (plus an optional byproduct `z`). The conditions must accept the same further positional (following the previously mentioned arguments) and keyword arguments as the forward mapping.


Why would the forward mapping return a byproduct `z` in addition to `y`?
It is mainly useful when the solution procedure creates objects such as Jacobians, which we want to reuse when computing or differentiating the conditions.
In that case, you may want to write the differentiation rules yourself for the conditions.
A more advanced application is given by [DifferentiableFrankWolfe.jl](https://github.com/gdalle/DifferentiableFrankWolfe.jl).

Keep in mind that derivatives of `z` will not be computed: the byproduct is considered constant during differentiation (unlike the case of multiple outputs outlined above).

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
