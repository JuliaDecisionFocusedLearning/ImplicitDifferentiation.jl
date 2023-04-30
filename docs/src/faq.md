# Frequently Asked Questions

## Higher-dimensional arrays

For simplicity, the examples only display functions that work on vectors.
However, arbitrary array sizes are supported.
Beware however, sparse arrays will be densified in the differentiation process.

## Multiple inputs / outputs

In this package, implicit functions can only take a single input array `x` and output a single output array `y` (plus the additional info `z`).
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

## Constrained optimization modeling

To express constrained optimization problems as implicit functions, you might need differentiable projections or proximal operators to write the optimality conditions.
See [_Efficient and modular implicit differentiation_](https://arxiv.org/abs/2105.15183) for precise formulations.

In case these operators are too complicated to code them yourself, here are a few places you can look:

- [MathOptSetDistances.jl](https://github.com/matbesancon/MathOptSetDistances.jl)
- [ProximalOperators.jl](https://github.com/JuliaFirstOrder/ProximalOperators.jl)

An alternative is differentiating through the KKT conditions, which is exactly what [DiffOpt.jl](https://github.com/jump-dev/DiffOpt.jl) does for JuMP models.

## Which autodiff backends are supported?

- Forward mode: ForwardDiff.jl
- Reverse mode: all the packages compatible with [ChainRules.jl](https://github.com/JuliaDiff/ChainRules.jl)

In the future, we would like to add [Enzyme.jl](https://github.com/EnzymeAD/Enzyme.jl) support.
