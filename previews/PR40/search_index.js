var documenterSearchIndex = {"docs":
[{"location":"api/#API-reference","page":"API reference","title":"API reference","text":"","category":"section"},{"location":"api/#Index","page":"API reference","title":"Index","text":"","category":"section"},{"location":"api/","page":"API reference","title":"API reference","text":"","category":"page"},{"location":"api/#Docstrings","page":"API reference","title":"Docstrings","text":"","category":"section"},{"location":"api/","page":"API reference","title":"API reference","text":"Modules = [ImplicitDifferentiation]","category":"page"},{"location":"api/#ImplicitDifferentiation.ImplicitFunction","page":"API reference","title":"ImplicitDifferentiation.ImplicitFunction","text":"ImplicitFunction{F,C,L}\n\nDifferentiable wrapper for an implicit function x -> y(x) whose output is defined by conditions F(x,y(x)) = 0.\n\nMore generally, we consider functions x -> (y(x),z(x)) and conditions F(x,y(x),z(x)) = 0, where z(x) contains additional information that is considered constant for differentiation purposes. Beware: the method zero(z) must exist.\n\nIf x ∈ ℝⁿ and y ∈ ℝᵈ, then we need as many conditions as output dimensions: F(x,y,z) ∈ ℝᵈ. Thanks to these conditions, we can compute the Jacobian of y(⋅) using the implicit function theorem:\n\n∂₂F(x,y(x),z(x)) * ∂y(x) = -∂₁F(x,y(x),z(x))\n\nThis amounts to solving a linear system A * J = -B, where A ∈ ℝᵈˣᵈ, B ∈ ℝᵈˣⁿ and J ∈ ℝᵈˣⁿ.\n\nFields:\n\nforward::F: callable of the form x -> (ŷ(x),z(x)).\nconditions::C: callable of the form (x,y,z) -> F(x,y,z)\nlinear_solver::L: callable of the form (A,b) -> u such that Au = b\n\n\n\n\n\n","category":"type"},{"location":"api/#ImplicitDifferentiation.ImplicitFunction-Tuple{Any, Any}","page":"API reference","title":"ImplicitDifferentiation.ImplicitFunction","text":"ImplicitFunction(forward, conditions)\n\nConstruct an ImplicitFunction{F,C,L} with Krylov.gmres as the default linear solver.\n\n\n\n\n\n","category":"method"},{"location":"api/#ImplicitDifferentiation.ImplicitFunction-Tuple{Any}","page":"API reference","title":"ImplicitDifferentiation.ImplicitFunction","text":"implicit(x[; kwargs...])\n\nMake ImplicitFunction{F,C,L} callable by applying implicit.forward.\n\n\n\n\n\n","category":"method"},{"location":"api/#ImplicitDifferentiation.LazyJacobianMul!","page":"API reference","title":"ImplicitDifferentiation.LazyJacobianMul!","text":"LazyJacobianMul!{M,N}\n\nCallable structure wrapping a lazy Jacobian operator with N-dimensional inputs into an in-place multiplication for vectors.\n\nFields\n\nJ::M: the lazy Jacobian of the function\ninput_size::NTuple{N,Int}: the array size of the function input\n\n\n\n\n\n","category":"type"},{"location":"api/#ImplicitDifferentiation.LazyJacobianTransposeMul!","page":"API reference","title":"ImplicitDifferentiation.LazyJacobianTransposeMul!","text":"LazyJacobianTransposeMul!{M,N}\n\nCallable structure wrapping a lazy Jacobian operator with N-dimensional outputs into an in-place multiplication for vectors.\n\nFields\n\nJ::M: the lazy Jacobian of the function\noutput_size::NTuple{N,Int}: the array size of the function output\n\n\n\n\n\n","category":"type"},{"location":"","page":"Home","title":"Home","text":"CurrentModule = ImplicitDifferentiation","category":"page"},{"location":"#ImplicitDifferentiation.jl","page":"Home","title":"ImplicitDifferentiation.jl","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"ImplicitDifferentiation.jl is a package for automatic differentiation of functions defined implicitly.","category":"page"},{"location":"#Background","page":"Home","title":"Background","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Implicit differentiation is useful to differentiate through two types of functions:","category":"page"},{"location":"","page":"Home","title":"Home","text":"Those for which automatic differentiation fails. Reasons can vary depending on your backend, but the most common include calls to external solvers, mutating operations or type restrictions.\nThose for which automatic differentiation is very slow. A common example is iterative procedures like fixed point equations or optimization algorithms.","category":"page"},{"location":"","page":"Home","title":"Home","text":"Please refer to Efficient and modular implicit differentiation for an introduction to the underlying theory.","category":"page"},{"location":"#Getting-started","page":"Home","title":"Getting started","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"To install the stable version, open a Julia REPL and run:","category":"page"},{"location":"","page":"Home","title":"Home","text":"julia> using Pkg; Pkg.add(\"ImplicitDifferentiation\")","category":"page"},{"location":"","page":"Home","title":"Home","text":"For the latest version, run this instead:","category":"page"},{"location":"","page":"Home","title":"Home","text":"julia> using Pkg; Pkg.add(url=\"https://github.com/gdalle/ImplicitDifferentiation.jl\")","category":"page"},{"location":"","page":"Home","title":"Home","text":"Check out the API reference to know more about the main object defined here, ImplicitFunction. The tutorials give you some ideas of real-life applications for our package.","category":"page"},{"location":"#Related-projects","page":"Home","title":"Related projects","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"DiffOpt.jl: differentiation of convex optimization problems\nInferOpt.jl: differentiation of combinatorial optimization problems\nNonconvexUtils.jl: contains the original implementation from which this package drew inspiration","category":"page"},{"location":"examples/0_basic/","page":"Basic use","title":"Basic use","text":"EditURL = \"https://github.com/gdalle/ImplicitDifferentiation.jl/blob/main/examples/0_basic.jl\"","category":"page"},{"location":"examples/0_basic/#Basic-use","page":"Basic use","title":"Basic use","text":"","category":"section"},{"location":"examples/0_basic/","page":"Basic use","title":"Basic use","text":"In this example, we demonstrate the basics of our package on a simple function that is not amenable to automatic differentiation.","category":"page"},{"location":"examples/0_basic/","page":"Basic use","title":"Basic use","text":"using ForwardDiff\nusing ImplicitDifferentiation\nusing LinearAlgebra\nusing Random\nusing Zygote\n\nRandom.seed!(63);\nnothing #hide","category":"page"},{"location":"examples/0_basic/#Autodiff-fails","page":"Basic use","title":"Autodiff fails","text":"","category":"section"},{"location":"examples/0_basic/","page":"Basic use","title":"Basic use","text":"ForwardDiff.jl and Zygote.jl are two prominent packages for automatic differentiation in Julia. While they are very generic, there are simple language constructs that they cannot differentiate through.","category":"page"},{"location":"examples/0_basic/","page":"Basic use","title":"Basic use","text":"function mysquare(x::AbstractArray)\n    a = [0.0]\n    a[1] = first(x)\n    return x .^ 2\nend;\nnothing #hide","category":"page"},{"location":"examples/0_basic/","page":"Basic use","title":"Basic use","text":"This is basically the componentwise square function but with an additional twist: a::Vector{Float64} is created internally, and its only element is replaced with the first element of x. We can check that it does what it's supposed to do.","category":"page"},{"location":"examples/0_basic/","page":"Basic use","title":"Basic use","text":"x = rand(2)\nmysquare(x) ≈ x .^ 2","category":"page"},{"location":"examples/0_basic/","page":"Basic use","title":"Basic use","text":"However, things start to go wrong when we compute Jacobians, due to the limitations of ForwardDiff.jl and Zygote.jl.","category":"page"},{"location":"examples/0_basic/","page":"Basic use","title":"Basic use","text":"try\n    ForwardDiff.jacobian(mysquare, x)\ncatch e\n    e\nend","category":"page"},{"location":"examples/0_basic/","page":"Basic use","title":"Basic use","text":"ForwardDiff.jl throws an error because it tries to call mysquare with an array of dual numbers, and cannot use one of these numbers to fill a (which has element type Float64).","category":"page"},{"location":"examples/0_basic/","page":"Basic use","title":"Basic use","text":"try\n    Zygote.jacobian(mysquare, x)\ncatch e\n    e\nend","category":"page"},{"location":"examples/0_basic/","page":"Basic use","title":"Basic use","text":"Zygote.jl also throws an error because it cannot handle mutation.","category":"page"},{"location":"examples/0_basic/#Implicit-functions","page":"Basic use","title":"Implicit functions","text":"","category":"section"},{"location":"examples/0_basic/","page":"Basic use","title":"Basic use","text":"The first possible use of ImplicitDifferentiation.jl is to overcome the limitations of automatic differentiation packages by defining Jacobians implicitly. Its main export is a type called ImplicitFunction, which we are going to see in action.","category":"page"},{"location":"examples/0_basic/","page":"Basic use","title":"Basic use","text":"First we define a forward pass correponding to the function we consider. It returns the actual output y of the function, as well as additional information z (which we don't need here, hence the 0). Importantly, this forward pass doesn't need to be differentiable.","category":"page"},{"location":"examples/0_basic/","page":"Basic use","title":"Basic use","text":"function forward(x)\n    y = mysquare(x)\n    z = 0\n    return y, z\nend","category":"page"},{"location":"examples/0_basic/","page":"Basic use","title":"Basic use","text":"Then we define conditions that the output y is supposed to satisfy. These conditions must be array-valued, with the same size as y. Here they are very obvious, but in later examples they will be more involved.","category":"page"},{"location":"examples/0_basic/","page":"Basic use","title":"Basic use","text":"function conditions(x, y, z)\n    c = y .- (x .^ 2)\n    return c\nend","category":"page"},{"location":"examples/0_basic/","page":"Basic use","title":"Basic use","text":"Finally, we construct a wrapper implicit around the previous objects. What does this wrapper do?","category":"page"},{"location":"examples/0_basic/","page":"Basic use","title":"Basic use","text":"implicit = ImplicitFunction(forward, conditions)","category":"page"},{"location":"examples/0_basic/","page":"Basic use","title":"Basic use","text":"When we call it as a function, it just falls back on implicit.forward, so unsurprisingly we get the same tuple (y, z).","category":"page"},{"location":"examples/0_basic/","page":"Basic use","title":"Basic use","text":"implicit(x)[1] ≈ x .^ 2","category":"page"},{"location":"examples/0_basic/#Autodiff-works","page":"Basic use","title":"Autodiff works","text":"","category":"section"},{"location":"examples/0_basic/","page":"Basic use","title":"Basic use","text":"And when we try to compute its Jacobian, the implicit function theorem is applied in the background to circumvent the lack of differentiablility of the forward pass.","category":"page"},{"location":"examples/0_basic/","page":"Basic use","title":"Basic use","text":"Now ForwardDiff.jl works seamlessly.","category":"page"},{"location":"examples/0_basic/","page":"Basic use","title":"Basic use","text":"ForwardDiff.jacobian(first ∘ implicit, x) ≈ Diagonal(2x)","category":"page"},{"location":"examples/0_basic/","page":"Basic use","title":"Basic use","text":"And so does Zygote.jl. Hurray!","category":"page"},{"location":"examples/0_basic/","page":"Basic use","title":"Basic use","text":"Zygote.jacobian(first ∘ implicit, x)[1] ≈ Diagonal(2x)","category":"page"},{"location":"examples/0_basic/","page":"Basic use","title":"Basic use","text":"We can even go higher-order by mixing the two packages (forward-over-reverse mode). The only technical requirement is to switch the linear solver to something that can handle dual numbers:","category":"page"},{"location":"examples/0_basic/","page":"Basic use","title":"Basic use","text":"linear_solver(A, b) = (Matrix(A) \\ b, (solved=true,))\nimplicit2 = ImplicitFunction(forward, conditions, linear_solver)","category":"page"},{"location":"examples/0_basic/","page":"Basic use","title":"Basic use","text":"Then the Jacobian itself is differentiable.","category":"page"},{"location":"examples/0_basic/","page":"Basic use","title":"Basic use","text":"h = rand(2)\nJ_Z(t) = Zygote.jacobian(first ∘ implicit2, x .+ t .* h)[1]\nForwardDiff.derivative(J_Z, 0) ≈ Diagonal(2h)","category":"page"},{"location":"examples/0_basic/","page":"Basic use","title":"Basic use","text":"","category":"page"},{"location":"examples/0_basic/","page":"Basic use","title":"Basic use","text":"This page was generated using Literate.jl.","category":"page"}]
}
