var documenterSearchIndex = {"docs":
[{"location":"examples/4_struct/","page":"Custom structs","title":"Custom structs","text":"EditURL = \"https://github.com/gdalle/ImplicitDifferentiation.jl/blob/main/test/4_struct.jl\"","category":"page"},{"location":"examples/4_struct/#Custom-structs","page":"Custom structs","title":"Custom structs","text":"","category":"section"},{"location":"examples/4_struct/","page":"Custom structs","title":"Custom structs","text":"In this example, we demonstrate implicit differentiation through functions that manipulate NamedTuples, as a first step towards compatibility with general structs.","category":"page"},{"location":"examples/4_struct/","page":"Custom structs","title":"Custom structs","text":"using ComponentArrays\nusing ImplicitDifferentiation\nusing Zygote","category":"page"},{"location":"examples/4_struct/#Implicit-function-wrapper","page":"Custom structs","title":"Implicit function wrapper","text":"","category":"section"},{"location":"examples/4_struct/","page":"Custom structs","title":"Custom structs","text":"We replicate a componentwise square function with NamedTuples, taking a=(x,y) as input and returning b=(u,v).","category":"page"},{"location":"examples/4_struct/","page":"Custom structs","title":"Custom structs","text":"forward(a::ComponentVector) = ComponentVector(u=a.x .^ 2, v=a.y .^ 2)\n\nfunction conditions(a::ComponentVector, b::ComponentVector)\n    return vcat(b.u .- a.x .^ 2, b.v .- a.y .^ 2)\nend\n\nimplicit = ImplicitFunction(forward, conditions);\nnothing #hide","category":"page"},{"location":"examples/4_struct/","page":"Custom structs","title":"Custom structs","text":"In order to be able to call Zygote.gradient, we use implicit to define a convoluted version of the squared Euclidean norm, which takes a ComponentVector as input and returns a real number.","category":"page"},{"location":"examples/4_struct/","page":"Custom structs","title":"Custom structs","text":"function fullnorm(a::ComponentVector)\n    b = implicit(a)\n    return sum(b)\nend","category":"page"},{"location":"examples/4_struct/#Testing","page":"Custom structs","title":"Testing","text":"","category":"section"},{"location":"examples/4_struct/","page":"Custom structs","title":"Custom structs","text":"a = ComponentVector(x=rand(3), y=rand(3))","category":"page"},{"location":"examples/4_struct/","page":"Custom structs","title":"Custom structs","text":"Let us first check that our weird squared norm returns the correct result.","category":"page"},{"location":"examples/4_struct/","page":"Custom structs","title":"Custom structs","text":"fullnorm(a) ≈ sum(abs2, a)","category":"page"},{"location":"examples/4_struct/","page":"Custom structs","title":"Custom structs","text":"Now we go one step further and compute its gradient, which involves the reverse rule for implicit.","category":"page"},{"location":"examples/4_struct/","page":"Custom structs","title":"Custom structs","text":"Zygote.gradient(fullnorm, a)[1] ≈ 2a","category":"page"},{"location":"examples/4_struct/","page":"Custom structs","title":"Custom structs","text":"","category":"page"},{"location":"examples/4_struct/","page":"Custom structs","title":"Custom structs","text":"This page was generated using Literate.jl.","category":"page"},{"location":"api/#API-reference","page":"API reference","title":"API reference","text":"","category":"section"},{"location":"api/#Index","page":"API reference","title":"Index","text":"","category":"section"},{"location":"api/","page":"API reference","title":"API reference","text":"","category":"page"},{"location":"api/#Docstrings","page":"API reference","title":"Docstrings","text":"","category":"section"},{"location":"api/","page":"API reference","title":"API reference","text":"Modules = [ImplicitDifferentiation]","category":"page"},{"location":"api/#ImplicitDifferentiation.ImplicitFunction","page":"API reference","title":"ImplicitDifferentiation.ImplicitFunction","text":"ImplicitFunction{F,C,L}\n\nDifferentiable wrapper for an implicit function x -> ŷ(x) whose output is defined by explicit conditions F(x,ŷ(x)) = 0.\n\nWe can obtain the Jacobian of ŷ with the implicit function theorem:\n\n∂₁F(x,ŷ(x)) + ∂₂F(x,ŷ(x)) * ∂ŷ(x) = 0\n\nIf x ∈ ℝⁿ, y ∈ ℝᵐ and F(x,y) ∈ ℝᶜ, this amounts to solving the linear system A * J = B, where A ∈ ℝᶜᵐ, B ∈ ℝᶜⁿ and J ∈ ℝᵐⁿ.\n\nFields:\n\nforward::F: callable of the form x -> ŷ(x)\nconditions::C: callable of the form (x,y) -> F(x,y)\nlinear_solver::L: callable of the form (A,b) -> u such that A * u = b\n\n\n\n\n\n","category":"type"},{"location":"api/#ImplicitDifferentiation.ImplicitFunction-Tuple{Any}","page":"API reference","title":"ImplicitDifferentiation.ImplicitFunction","text":"implicit(x)\n\nMake ImplicitFunction callable by applying implicit.forward.\n\n\n\n\n\n","category":"method"},{"location":"api/#ImplicitDifferentiation.ImplicitFunction-Union{Tuple{C}, Tuple{F}, Tuple{F, C}} where {F, C}","page":"API reference","title":"ImplicitDifferentiation.ImplicitFunction","text":"ImplicitFunction(forward, conditions)\n\nConstruct an ImplicitFunction with Krylov.gmres as the default linear solver.\n\nSee also\n\nImplicitFunction{F,C,L}\n\n\n\n\n\n","category":"method"},{"location":"api/#ChainRulesCore.frule-Tuple{ChainRulesCore.RuleConfig, Any, ImplicitFunction, AbstractVector}","page":"API reference","title":"ChainRulesCore.frule","text":"frule(rc, (_, dx), implicit, x)\n\nCustom forward rule for ImplicitFunction.\n\nWe compute the Jacobian-vector product Jv by solving Au = Bv and setting Jv = u.\n\n\n\n\n\n","category":"method"},{"location":"api/#ChainRulesCore.rrule-Tuple{ChainRulesCore.RuleConfig, ImplicitFunction, AbstractVector}","page":"API reference","title":"ChainRulesCore.rrule","text":"rrule(rc, implicit, x)\n\nCustom reverse rule for ImplicitFunction.\n\nWe compute the vector-Jacobian product Jᵀv by solving Aᵀu = v and setting Jᵀv = Bᵀu.\n\n\n\n\n\n","category":"method"},{"location":"api/#ChainRulesCore.rrule-Tuple{typeof(simplex_projection), AbstractVector{<:Real}}","page":"API reference","title":"ChainRulesCore.rrule","text":"rrule(::typeof(simplex_projection), z)\n\nCustom reverse rule for simplex_projection which bypasses the sorting step.\n\nSee https://arxiv.org/abs/1602.02068 for details.\n\n\n\n\n\n","category":"method"},{"location":"api/#ImplicitDifferentiation.simplex_projection-Tuple{AbstractVector{<:Real}}","page":"API reference","title":"ImplicitDifferentiation.simplex_projection","text":"simplex_projection(z)\n\nCompute the Euclidean projection onto the probability simplex.\n\n\n\n\n\n","category":"method"},{"location":"api/#ImplicitDifferentiation.simplex_projection_and_support-Union{Tuple{AbstractVector{R}}, Tuple{R}} where R<:Real","page":"API reference","title":"ImplicitDifferentiation.simplex_projection_and_support","text":"simplex_projection_and_support(z)\n\nCompute the Euclidean projection onto the probability simplex and the set of indices where it is nonzero.\n\nSee https://arxiv.org/abs/1602.02068 for details.\n\n\n\n\n\n","category":"method"},{"location":"background/#Mathematical-background","page":"Mathematical background","title":"Mathematical background","text":"","category":"section"},{"location":"background/","page":"Mathematical background","title":"Mathematical background","text":"warning: Work in progress\nIn the meantime, please refer to the preprint Efficient and modular implicit differentiation for an introduction to the methods implemented here.","category":"page"},{"location":"examples/2_constrained_optimization/","page":"Constrained optimization","title":"Constrained optimization","text":"EditURL = \"https://github.com/gdalle/ImplicitDifferentiation.jl/blob/main/test/2_constrained_optimization.jl\"","category":"page"},{"location":"examples/2_constrained_optimization/#Constrained-optimization","page":"Constrained optimization","title":"Constrained optimization","text":"","category":"section"},{"location":"examples/2_constrained_optimization/","page":"Constrained optimization","title":"Constrained optimization","text":"In this example, we show how to differentiate through the solution of the following constrained optimization problem:","category":"page"},{"location":"examples/2_constrained_optimization/","page":"Constrained optimization","title":"Constrained optimization","text":"haty(x) = min_y in mathcalC f(x y)","category":"page"},{"location":"examples/2_constrained_optimization/","page":"Constrained optimization","title":"Constrained optimization","text":"where mathcalC is a closed convex set. The optimal solution can be found as the fixed point of the projected gradient algorithm for any step size eta. This insight yields the following optimality conditions:","category":"page"},{"location":"examples/2_constrained_optimization/","page":"Constrained optimization","title":"Constrained optimization","text":"F(x haty(x)) = 0 quad textwith quad F(xy) = mathrmproj_mathcalC(y - eta nabla_2 f(x y)) - y","category":"page"},{"location":"examples/2_constrained_optimization/","page":"Constrained optimization","title":"Constrained optimization","text":"using ImplicitDifferentiation\nusing Ipopt\nusing JuMP\nusing Zygote","category":"page"},{"location":"examples/2_constrained_optimization/#Projecting-onto-the-simplex","page":"Constrained optimization","title":"Projecting onto the simplex","text":"","category":"section"},{"location":"examples/2_constrained_optimization/","page":"Constrained optimization","title":"Constrained optimization","text":"We focus on minimizing f(xy) = lVert x - y rVert_2^2. We also assume that mathcalC = Delta^n is the n-dimensional probability simplex, because we know an exact procedure to compute the projection and its Jacobian.","category":"page"},{"location":"examples/2_constrained_optimization/","page":"Constrained optimization","title":"Constrained optimization","text":"Both of these procedures are outlined in From Softmax to Sparsemax: A Sparse Model of Attention and Multi-Label Classification. You can find an implementation in the function simplex_projection. Because this function involves a call to sort, standard AD backends cannot differentiate through it, which is why we also had to define a chain rule for it.","category":"page"},{"location":"examples/2_constrained_optimization/#Implicit-function-wrapper","page":"Constrained optimization","title":"Implicit function wrapper","text":"","category":"section"},{"location":"examples/2_constrained_optimization/","page":"Constrained optimization","title":"Constrained optimization","text":"We now wrap a black box optimizer inside an ImplicitFunction to compare its implicit differentiation with the explicit procedure given above.","category":"page"},{"location":"examples/2_constrained_optimization/","page":"Constrained optimization","title":"Constrained optimization","text":"function forward(x)\n    n = length(x)\n    optimizer = optimizer_with_attributes(Ipopt.Optimizer, \"print_level\" => 0)\n    model = Model(optimizer)\n    @variable(model, y[1:n] >= 0)\n    @constraint(model, sum(y) == 1)\n    @objective(model, Min, sum((y .- x) .^ 2))\n    optimize!(model)\n    return value.(y)\nend;\n\nconditions(x, y) = simplex_projection(y - 0.1 * 2(y - x)) - y;\n\nimplicit = ImplicitFunction(forward, conditions);\nnothing #hide","category":"page"},{"location":"examples/2_constrained_optimization/#Testing","page":"Constrained optimization","title":"Testing","text":"","category":"section"},{"location":"examples/2_constrained_optimization/","page":"Constrained optimization","title":"Constrained optimization","text":"Let us study the behavior of our implicit function.","category":"page"},{"location":"examples/2_constrained_optimization/","page":"Constrained optimization","title":"Constrained optimization","text":"x = rand(5)","category":"page"},{"location":"examples/2_constrained_optimization/","page":"Constrained optimization","title":"Constrained optimization","text":"We can see that the forward pass computes the projection correctly, at least up to numerical precision.","category":"page"},{"location":"examples/2_constrained_optimization/","page":"Constrained optimization","title":"Constrained optimization","text":"hcat(simplex_projection(x), implicit(x))","category":"page"},{"location":"examples/2_constrained_optimization/","page":"Constrained optimization","title":"Constrained optimization","text":"And the same goes for the Jacobian.","category":"page"},{"location":"examples/2_constrained_optimization/","page":"Constrained optimization","title":"Constrained optimization","text":"cat(Zygote.jacobian(simplex_projection, x)[1], Zygote.jacobian(implicit, x)[1]; dims=3)","category":"page"},{"location":"examples/2_constrained_optimization/","page":"Constrained optimization","title":"Constrained optimization","text":"","category":"page"},{"location":"examples/2_constrained_optimization/","page":"Constrained optimization","title":"Constrained optimization","text":"This page was generated using Literate.jl.","category":"page"},{"location":"","page":"Home","title":"Home","text":"CurrentModule = ImplicitDifferentiation","category":"page"},{"location":"#ImplicitDifferentiation.jl","page":"Home","title":"ImplicitDifferentiation.jl","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"ImplicitDifferentiation.jl is a package for automatic differentiation of implicit functions.","category":"page"},{"location":"#Getting-started","page":"Home","title":"Getting started","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"To install it, open a Julia Pkg REPL and run:","category":"page"},{"location":"","page":"Home","title":"Home","text":"pkg> add \"https://github.com/gdalle/ImplicitDifferentiation.jl\"","category":"page"},{"location":"#Related-packages","page":"Home","title":"Related packages","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"DiffOpt.jl: differentiation of convex optimization problems","category":"page"},{"location":"examples/1_unconstrained_optimization/","page":"Unconstrained optimization","title":"Unconstrained optimization","text":"EditURL = \"https://github.com/gdalle/ImplicitDifferentiation.jl/blob/main/test/1_unconstrained_optimization.jl\"","category":"page"},{"location":"examples/1_unconstrained_optimization/#Unconstrained-optimization","page":"Unconstrained optimization","title":"Unconstrained optimization","text":"","category":"section"},{"location":"examples/1_unconstrained_optimization/","page":"Unconstrained optimization","title":"Unconstrained optimization","text":"In this example, we show how to differentiate through the solution of the following unconstrained optimization problem:","category":"page"},{"location":"examples/1_unconstrained_optimization/","page":"Unconstrained optimization","title":"Unconstrained optimization","text":"haty(x) = min_y in mathbbR^m f(x y)","category":"page"},{"location":"examples/1_unconstrained_optimization/","page":"Unconstrained optimization","title":"Unconstrained optimization","text":"The optimality conditions are given by gradient stationarity:","category":"page"},{"location":"examples/1_unconstrained_optimization/","page":"Unconstrained optimization","title":"Unconstrained optimization","text":"F(x haty(x)) = 0 quad textwith quad F(xy) = nabla_2 f(x y) = 0","category":"page"},{"location":"examples/1_unconstrained_optimization/","page":"Unconstrained optimization","title":"Unconstrained optimization","text":"using ImplicitDifferentiation\nusing Optim: optimize, minimizer, LBFGS\nusing Zygote","category":"page"},{"location":"examples/1_unconstrained_optimization/#Implicit-function-wrapper","page":"Unconstrained optimization","title":"Implicit function wrapper","text":"","category":"section"},{"location":"examples/1_unconstrained_optimization/","page":"Unconstrained optimization","title":"Unconstrained optimization","text":"To make verification easy, we minimize a quadratic objective","category":"page"},{"location":"examples/1_unconstrained_optimization/","page":"Unconstrained optimization","title":"Unconstrained optimization","text":"f(x y) = lVert y - x rVert^2","category":"page"},{"location":"examples/1_unconstrained_optimization/","page":"Unconstrained optimization","title":"Unconstrained optimization","text":"In this case, the optimization algorithm is very simple, but still we can implement it as a black box to show that it doesn't change the result.","category":"page"},{"location":"examples/1_unconstrained_optimization/","page":"Unconstrained optimization","title":"Unconstrained optimization","text":"function forward(x)\n    f(y) = sum(abs2, y-x)\n    y0 = zero(x)\n    res = optimize(f, y0, LBFGS(); autodiff=:forward)\n    y = minimizer(res)\n    return y\nend;\nnothing #hide","category":"page"},{"location":"examples/1_unconstrained_optimization/","page":"Unconstrained optimization","title":"Unconstrained optimization","text":"On the other hand, optimality conditions should be provided explicitly whenever possible, so as to avoid nesting automatic differentiation calls.","category":"page"},{"location":"examples/1_unconstrained_optimization/","page":"Unconstrained optimization","title":"Unconstrained optimization","text":"conditions(x, y) = 2(y - x);\nnothing #hide","category":"page"},{"location":"examples/1_unconstrained_optimization/","page":"Unconstrained optimization","title":"Unconstrained optimization","text":"We now have all the ingredients to construct our implicit function.","category":"page"},{"location":"examples/1_unconstrained_optimization/","page":"Unconstrained optimization","title":"Unconstrained optimization","text":"implicit = ImplicitFunction(forward, conditions);\nnothing #hide","category":"page"},{"location":"examples/1_unconstrained_optimization/#Testing","page":"Unconstrained optimization","title":"Testing","text":"","category":"section"},{"location":"examples/1_unconstrained_optimization/","page":"Unconstrained optimization","title":"Unconstrained optimization","text":"x = rand(5)","category":"page"},{"location":"examples/1_unconstrained_optimization/","page":"Unconstrained optimization","title":"Unconstrained optimization","text":"Let's start by taking a look at the forward pass, which should be the identity function.","category":"page"},{"location":"examples/1_unconstrained_optimization/","page":"Unconstrained optimization","title":"Unconstrained optimization","text":"implicit(x)","category":"page"},{"location":"examples/1_unconstrained_optimization/","page":"Unconstrained optimization","title":"Unconstrained optimization","text":"We now check whether the behavior of our ImplicitFunction wrapper is coherent with the theoretical derivatives.","category":"page"},{"location":"examples/1_unconstrained_optimization/","page":"Unconstrained optimization","title":"Unconstrained optimization","text":"Zygote.jacobian(implicit, x)[1]","category":"page"},{"location":"examples/1_unconstrained_optimization/","page":"Unconstrained optimization","title":"Unconstrained optimization","text":"As expected, we recover the identity matrix as Jacobian.","category":"page"},{"location":"examples/1_unconstrained_optimization/","page":"Unconstrained optimization","title":"Unconstrained optimization","text":"","category":"page"},{"location":"examples/1_unconstrained_optimization/","page":"Unconstrained optimization","title":"Unconstrained optimization","text":"This page was generated using Literate.jl.","category":"page"},{"location":"examples/3_optimal_transport/","page":"Optimal transport","title":"Optimal transport","text":"EditURL = \"https://github.com/gdalle/ImplicitDifferentiation.jl/blob/main/test/3_optimal_transport.jl\"","category":"page"},{"location":"examples/3_optimal_transport/#Optimal-transport","page":"Optimal transport","title":"Optimal transport","text":"","category":"section"},{"location":"examples/3_optimal_transport/","page":"Optimal transport","title":"Optimal transport","text":"In this example, we show how to differentiate through the solution of the entropy-regularized optimal transport problem.","category":"page"},{"location":"examples/3_optimal_transport/","page":"Optimal transport","title":"Optimal transport","text":"using Distances\nusing FiniteDiff\nusing ImplicitDifferentiation\nusing OptimalTransport\nusing Zygote","category":"page"},{"location":"examples/3_optimal_transport/#Introduction","page":"Optimal transport","title":"Introduction","text":"","category":"section"},{"location":"examples/3_optimal_transport/","page":"Optimal transport","title":"Optimal transport","text":"Here we give a brief introduction to optimal transport, see the book by Gabriel Peyré and Marco Cuturi for more details.","category":"page"},{"location":"examples/3_optimal_transport/#Problem-description","page":"Optimal transport","title":"Problem description","text":"","category":"section"},{"location":"examples/3_optimal_transport/","page":"Optimal transport","title":"Optimal transport","text":"Suppose we have a distribution of mass mu in Delta^n over points x_1  x_n in mathbbR^d (where Delta denotes the probability simplex). We want to transport it to a distribution nu in Delta^m over points y_1  y_m in mathbbR^d. The unit moving cost from x to y is proportional to the squared Euclidean distance c(x y) = lVert x - y rVert_2^2.","category":"page"},{"location":"examples/3_optimal_transport/","page":"Optimal transport","title":"Optimal transport","text":"A transportation plan can be described by a coupling p = Pi(mu nu), i.e. a probability distribution on the product space with the right marginals:","category":"page"},{"location":"examples/3_optimal_transport/","page":"Optimal transport","title":"Optimal transport","text":"Pi(mu nu) = p in Delta^n times m p mathbf1 = mu p^top mathbf1 = nu","category":"page"},{"location":"examples/3_optimal_transport/","page":"Optimal transport","title":"Optimal transport","text":"Let C in mathbbR^n times m be the moving cost matrix, with C_ij = c(x_i y_j). The basic optimization problem we want to solve is a linear program:","category":"page"},{"location":"examples/3_optimal_transport/","page":"Optimal transport","title":"Optimal transport","text":"hatp(C) = min_p in Pi(mu nu) sum_ij p_ij C_ij","category":"page"},{"location":"examples/3_optimal_transport/","page":"Optimal transport","title":"Optimal transport","text":"In order to make it smoother, we add an entropic regularization term:","category":"page"},{"location":"examples/3_optimal_transport/","page":"Optimal transport","title":"Optimal transport","text":"hatp_varepsilon(C) = min_p in Pi(mu nu) sum_ij left(p_ij C_ij + varepsilon p_ij log fracp_ijmu_i nu_j right)","category":"page"},{"location":"examples/3_optimal_transport/#Sinkhorn-algorithm","page":"Optimal transport","title":"Sinkhorn algorithm","text":"","category":"section"},{"location":"examples/3_optimal_transport/","page":"Optimal transport","title":"Optimal transport","text":"To solve the regularized problem, we can use the Sinkhorn fixed point algorithm. Let K in mathbbR^n times m be the matrix defined by K_ij = exp(-C_ij  varepsilon). Then the optimal coupling hatp_varepsilon(C) can be written as:","category":"page"},{"location":"examples/3_optimal_transport/","page":"Optimal transport","title":"Optimal transport","text":"hatp_varepsilon(C) = mathrmdiag(hatu)  K  mathrmdiag(hatv) tag1","category":"page"},{"location":"examples/3_optimal_transport/","page":"Optimal transport","title":"Optimal transport","text":"where hatu and hatv are the fixed points of the following Sinkhorn fixed point iteration:","category":"page"},{"location":"examples/3_optimal_transport/","page":"Optimal transport","title":"Optimal transport","text":"u^t+1 = fracmuKv^t qquad textand qquad v^t+1 = fracnuK^top u^t","category":"page"},{"location":"examples/3_optimal_transport/","page":"Optimal transport","title":"Optimal transport","text":"The implicit function theorem can be used to differentiate hatu and hatv with respect to C, mu and/or nu. This can be combined with automatic differentiation of Equation (1) to find the Jacobian","category":"page"},{"location":"examples/3_optimal_transport/","page":"Optimal transport","title":"Optimal transport","text":"J = fracpartial  mathrmvec(hatp_varepsilon(C))partial  mathrmvec(C)","category":"page"},{"location":"examples/3_optimal_transport/#Implicit-function-wrapper","page":"Optimal transport","title":"Implicit function wrapper","text":"","category":"section"},{"location":"examples/3_optimal_transport/","page":"Optimal transport","title":"Optimal transport","text":"For now, ImplicitFunction objects do not take multiple arguments, so we use non-constant global variables instead (even though we shouldn't)","category":"page"},{"location":"examples/3_optimal_transport/","page":"Optimal transport","title":"Optimal transport","text":"d = 10\nn = 3\nm = 4\n\nX = rand(d, n)\nY = rand(d, m)\n\nμ = fill(1 / n, n)\nν = fill(1 / m, m)\nC_vec = vec(pairwise(SqEuclidean(), X, Y, dims=2))\n\nε = 1.0;\nnothing #hide","category":"page"},{"location":"examples/3_optimal_transport/","page":"Optimal transport","title":"Optimal transport","text":"We now embed the Sinkhorn algorithm and optimality conditions inside an ImplicitFunction struct. For technical reasons related to optimality checking, our forward procedure returns hatu instead of hatp_varepsilon.","category":"page"},{"location":"examples/3_optimal_transport/","page":"Optimal transport","title":"Optimal transport","text":"function forward(C_vec)\n    C = reshape(C_vec, n, m)\n    solver = OptimalTransport.build_solver(μ, ν, C, ε, SinkhornGibbs())\n    OptimalTransport.solve!(solver)\n    û = solver.cache.u\n    return û\nend\n\nfunction conditions(C_vec, û)\n    C = reshape(C_vec, n, m)\n    K = exp.(.-C ./ ε)\n    v̂ = ν ./ (K' * û)\n    return û .- μ ./ (K * v̂)\nend\n\nimplicit = ImplicitFunction(forward, conditions);\nnothing #hide","category":"page"},{"location":"examples/3_optimal_transport/#Testing","page":"Optimal transport","title":"Testing","text":"","category":"section"},{"location":"examples/3_optimal_transport/","page":"Optimal transport","title":"Optimal transport","text":"First, let us check that the forward pass works correctly","category":"page"},{"location":"examples/3_optimal_transport/","page":"Optimal transport","title":"Optimal transport","text":"û, _ = forward(C_vec)\n\nmaximum(abs, conditions(C_vec, û))","category":"page"},{"location":"examples/3_optimal_transport/","page":"Optimal transport","title":"Optimal transport","text":"Using the implicit function defined above, we can build an autodiff-compatible implementation of transportation_plan which does not require backpropagating through the Sinkhorn iterations:","category":"page"},{"location":"examples/3_optimal_transport/","page":"Optimal transport","title":"Optimal transport","text":"function transportation_plan(C_vec)\n    C = reshape(C_vec, n, m)\n    K = exp.(.-C ./ ε)\n    û = implicit(C_vec)\n    v̂ = ν ./ (K' * û)\n    p̂_vec = vec(û .* K .* v̂')\n    return p̂_vec\nend;\nnothing #hide","category":"page"},{"location":"examples/3_optimal_transport/","page":"Optimal transport","title":"Optimal transport","text":"Let us compare with the result obtained using finite differences:","category":"page"},{"location":"examples/3_optimal_transport/","page":"Optimal transport","title":"Optimal transport","text":"J_autodiff = Zygote.jacobian(transportation_plan, C_vec)[1]\nJ_finitediff = FiniteDiff.finite_difference_jacobian(transportation_plan, C_vec)\nmaximum(abs, J_autodiff - J_finitediff)","category":"page"},{"location":"examples/3_optimal_transport/","page":"Optimal transport","title":"Optimal transport","text":"","category":"page"},{"location":"examples/3_optimal_transport/","page":"Optimal transport","title":"Optimal transport","text":"This page was generated using Literate.jl.","category":"page"}]
}