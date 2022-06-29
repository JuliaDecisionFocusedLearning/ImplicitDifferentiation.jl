```@meta
CurrentModule = ImplicitDifferentiation
```

# ImplicitDifferentiation.jl

[ImplicitDifferentiation.jl](https://github.com/gdalle/ImplicitDifferentiation.jl) is a package for automatic differentiation of implicit functions.

## Getting started

To install the stable version, open a Julia REPL and run:
```julia
julia> using Pkg; Pkg.add("ImplicitDifferentiation")
```
For the latest version, run this instead:
```julia
julia> using Pkg; Pkg.add(url="https://github.com/gdalle/ImplicitDifferentiation.jl")
```

## Related packages

- [DiffOpt.jl](https://github.com/jump-dev/DiffOpt.jl): differentiation of convex optimization problems
- [InferOpt.jl](https://github.com/axelparmentier/InferOpt.jl): differentiation of combinatorial optimization problems
- [NonconvexUtils.jl](https://github.com/JuliaNonconvex/NonconvexUtils.jl): contains the original implementation from which this package drew inspiration