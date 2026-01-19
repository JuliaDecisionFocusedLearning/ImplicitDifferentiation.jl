# FAQ

## Supported autodiff backends

To differentiate through an `ImplicitFunction`, the following backends are supported.

| Backend                                                                | Forward mode | Reverse mode |
| :--------------------------------------------------------------------- | :----------- | :----------- |
| [ForwardDiff.jl](https://github.com/JuliaDiff/ForwardDiff.jl)          | yes          | -            |
| [ChainRules.jl](https://github.com/JuliaDiff/ChainRules.jl)-compatible | no           | yes          |
| [Enzyme.jl](https://github.com/EnzymeAD/Enzyme.jl)                     | soon         | soon         |

By default, the conditions are differentiated using the same "outer" backend that is trying to differentiate the `ImplicitFunction`.
However, this can be switched to any other "inner" backend compatible with [DifferentiationInterface.jl](https://github.com/gdalle/DifferentiationInterface.jl) (i.e. a subtype of `ADTypes.AbstractADType`).

## Input and output types

### Arrays

Functions that eat or spit out arbitrary arrays are supported, as long as the forward mapping _and_ conditions return arrays of the same size.
The array types involved should be mutable.

### Scalars

Functions that eat or spit out a single number are not supported.
The forward mapping _and_ conditions need vectors: instead of returning `val` you should return `[val]` (a 1-element `Vector`).

## Number of inputs and outputs

Most of the documentation is written for the simple case where the forward mapping is `x -> y`, i.e. one input and one output.
What can you do to handle multiple inputs or outputs?
Well, it depends whether you want their derivatives or not.

|                      | Derivatives needed                      | Derivatives not needed                          |
| :------------------- | :-------------------------------------- | :---------------------------------------------- |
| **Multiple inputs**  | Make `x` a `ComponentVector`            | Supply `args...` to `forward`                   |
| **Multiple outputs** | Make `y` and `c` two `ComponentVector`s | Let `forward` return a nontrivial byproduct `z` |

We now detail each of these options.

### Multiple inputs or outputs | Derivatives needed

Say your forward mapping takes multiple inputs and returns multiple outputs, such that you want derivatives for all of them.

The trick is to leverage [ComponentArrays.jl](https://github.com/jonniedie/ComponentArrays.jl) to wrap all the inputs inside a single a `ComponentVector`, and do the same for all the outputs.
See the examples for a demonstration.

!!! warning
    You may run into issues trying to differentiate through the `ComponentVector` constructor.
    For instance, Zygote.jl will throw `ERROR: Mutating arrays is not supported`.
    Check out [this issue](https://github.com/gdalle/ImplicitDifferentiation.jl/issues/67) for a dirty workaround involving custom chain rules for the constructor.

### Multiple inputs | Derivatives not needed

If your forward mapping (or conditions) takes multiple inputs but you don't care about derivatives, then you can add further positional arguments beyond `x`.
It is important to make sure that the forward mapping and conditions accept the same set of arguments, even if each of these functions only uses a subset of them.

```julia
forward(x, arg1, arg2) = y, z
conditions(x, y, z, arg1, arg2) = c
```

All of the positional arguments apart from `x` will get zero tangents during differentiation of the implicit function.

### Multiple outputs | Derivatives not needed

The last and most tricky situation is when your forward mapping returns multiple outputs, but you only care about some of their derivatives.
Then, you need to group the objects you don't want to differentiate into a nontrivial "byproduct" `z`, returned alongside the actual output `y`.
This way, derivatives of `z` will not be computed: the byproduct is considered constant during differentiation.

This is mainly useful when the solution procedure creates objects such as Jacobians, which we want to reuse when computing or differentiating the conditions.
In that case, you may want to write the conditions differentiation rules yourself.
A more advanced application is given by [DifferentiableFrankWolfe.jl](https://github.com/gdalle/DifferentiableFrankWolfe.jl).

## Modeling tips

### Writing conditions

We recommend that the conditions themselves do not involve calls to autodiff, even when they describe a gradient.
Otherwise, you will need to make sure that nested autodiff works well in your case (i.e. that the "outer" backend can differentiate through the "inner" backend).
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
@memoize Dict forward(x, args...; kwargs...) = y, z
```

### Linear solver selection

Differentiating your implicit function requires to solve a linear system. By default, an iterative solver (see [`IterativeSolver`](@ref)) combined with a matrix-free representation of the jacobian (see [`OperatorRepresentation`](@ref)) is used. You can change the linear solver using the `linear_solver` keyword argument of the `ImplicitFunction` constructor, choosing between:

- [`IterativeSolver`](@ref);
- [`IterativeLeastSquaresSolver`](@ref);
- [`DirectLinearSolver`](@ref).

Keyword arguments can be passed to the constructors of `IterativeSolver` and `IterativeLeastSquaresSolver`, they will be forwarded to the `KrylovKit.linsolve` and `KrylovKit.lssolve` functions, respectively.

Note that for the `DirectLinearSolver`, you must switch to a [`MatrixRepresentation`](@ref) using the `representation` argument : `ImplicitFunction(forward, conditions; linear_solver = DirectLinearSolver(), representation = MatrixRepresentation())`.
