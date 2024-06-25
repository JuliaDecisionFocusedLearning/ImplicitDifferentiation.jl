# ImplicitDifferentiation.jl

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://JuliaDecisionFocusedLearning.github.io/ImplicitDifferentiation.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://JuliaDecisionFocusedLearning.github.io/ImplicitDifferentiation.jl/dev/)
[![Build Status](https://github.com/JuliaDecisionFocusedLearning/ImplicitDifferentiation.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/JuliaDecisionFocusedLearning/ImplicitDifferentiation.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/JuliaDecisionFocusedLearning/ImplicitDifferentiation.jl/branch/main/graph/badge.svg)](https://app.codecov.io/gh/JuliaDecisionFocusedLearning/ImplicitDifferentiation.jl)
[![Code Style: Blue](https://img.shields.io/badge/code%20style-blue-4495d1.svg)](https://github.com/invenia/BlueStyle)
[![Aqua QA](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)

[ImplicitDifferentiation.jl](https://github.com/JuliaDecisionFocusedLearning/ImplicitDifferentiation.jl) is a package for automatic differentiation of functions defined implicitly, i.e., _forward mappings_

```math
x \in \mathbb{R}^n \longmapsto y(x) \in \mathbb{R}^m
```

whose output is defined by _conditions_

```math
c(x,y(x)) = 0 \in \mathbb{R}^m
```

## Background

Implicit differentiation is useful to differentiate through two types of functions:

- Those for which automatic differentiation fails. Reasons can vary depending on your backend, but the most common include calls to external solvers, mutating operations or type restrictions.
- Those for which automatic differentiation is very slow. A common example is iterative procedures like fixed point equations or optimization algorithms.

If you just need a quick overview, check out our [JuliaCon 2022 talk](https://www.youtube.com/watch?v=TkVDcujVNJ4&feature=youtu.be).
If you want a deeper dive into the theory, you can refer to the paper [_Efficient and modular implicit differentiation_](https://papers.nips.cc/paper_files/paper/2022/hash/228b9279ecf9bbafe582406850c57115-Abstract-Conference.html) by Blondel et al. (2022).

## Getting started

To install the stable version, open a Julia REPL and run:

```julia
using Pkg; Pkg.add("ImplicitDifferentiation")
```

For the latest version, run this instead:

```julia
using Pkg; Pkg.add(url="https://github.com/JuliaDecisionFocusedLearning/ImplicitDifferentiation.jl")
```

Please read the [documentation](https://JuliaDecisionFocusedLearning.github.io/ImplicitDifferentiation.jl/stable/), especially the examples and FAQ.

## Related projects

In Julia:

- [SciML](https://sciml.ai/) ecosystem, especially [LinearSolve.jl](https://github.com/SciML/LinearSolve.jl), [NonlinearSolve.jl](https://github.com/SciML/NonlinearSolve.jl) and [Optimization.jl](https://github.com/SciML/Optimization.jl)
- [jump-dev/DiffOpt.jl](https://github.com/jump-dev/DiffOpt.jl): differentiation of convex optimization problems
- [axelparmentier/InferOpt.jl](https://github.com/axelparmentier/InferOpt.jl): approximate differentiation of combinatorial optimization problems
- [JuliaNonconvex/NonconvexUtils.jl](https://github.com/JuliaNonconvex/NonconvexUtils.jl): contains the original implementation from which this package drew inspiration

In Python:

- [google/jaxopt](https://github.com/google/jaxopt): hardware accelerated, batchable and differentiable optimizers in JAX