# # Gallery

# ## Distance to the unit circle

#=
We show how to compute a differentiable distance field to the unit circle. Note that, for this specific case,
- an analytical solution is available ($$d(x) = ||x||-1$$), allowing some tests;
- we could directly write an `Optim` function returning the distance function, and this function is differentiable.
=#
using ImplicitDifferentiation
using LinearAlgebra
using Optim
using StaticArrays
using ForwardDiff

""" Find `θ0` such that the squared distance between `x` and the point `x_circle(θ0)` on the unit circle is minimal. """
function forward_unit_circle(x)
    function f(θ)
        _θ = first(θ)
        u = x .- SA[cos(_θ), sin(_θ)]
        return sum(abs2, u)
    end
    θ_init = -π # pretend we don't know the solution
    results = optimize(f, [θ_init])

    y = Optim.minimizer(results) # vector of length 1
    z = nothing

    return y, z
end

""" Gradient, with respect to `θ0` (represented here by `y`) of the squared distance between `x` and `x_circle(θ0)`. """
function conditions(x, y, _z)
    x1, x2 = x
    θ = first(y)
    c = cos(θ)
    s = sin(θ)
    ∇₂f = 2 .* [s * (x1 - c) + (-c) * (x2 - s)]
    return ∇₂f
end

#=
Now, let's define three different functions for our distance field:
- one using directly the forward function : `d_explicit`;
- one using an `ImplicitFunction` with an `IterativeLinearSolver`: `d_impl_iter`;
- one using an `ImplicitFunction` with the `DirectLinearSolver`: `d_impl_direct`.
=#
function d_forward(x)
    θ = first(first(forward_unit_circle(x)))
    dist = norm(x .- SA[cos(θ), sin(θ)])
    return dist
end

impl_iter = ImplicitFunction(forward_unit_circle, conditions;)
function d_implicit_iter(x)
    θ = first(first(impl_iter(x)))
    dist = norm(x .- SA[cos(θ), sin(θ)])
    return dist
end

impl_direct = ImplicitFunction(
    forward_unit_circle,
    conditions;
    representation=MatrixRepresentation(),
    linear_solver=DirectLinearSolver(),
)
function d_impl_direct(x)
    θ = first(first(impl_direct(x)))
    dist = norm(x .- SA[cos(θ), sin(θ)])
    return dist
end

#=
Evaluated on a point, let's say, `x = [1.1, 0.0]`, they all return the same result (`d(x) ≈ 0.1`):
=#
p = [1.1, 0.0]
@show d_forward(p)
@show d_implicit_iter(p)
@show d_impl_direct(p)
@test d_forward(p) ≈ d_implicit_iter(p) ≈ d_impl_direct(p) #src

#=
The most interesting part is the differentiation of two implicit functions. Note that the gradient of the
distance to the unit circle is the following unit vector:
```math
\nabla d(x) = \dfrac{1}{\lVert x \rVert} x.
```
=#
ν_iter = ForwardDiff.gradient(d_implicit_iter, p)
@show ν_iter, norm(ν_iter)
@test abs(ν_iter[1] - 1.0) < eps(1.0) # src
@test abs(ν_iter[2]) < eps(0.0) # src
ν_direct = ForwardDiff.gradient(d_impl_direct, p)
@show ν_direct, norm(ν_direct)
@test abs(ν_direct[1] - 1.0) < eps(1.0) # src
@test abs(ν_direct[2]) < eps(0.0) # src

#=
Check for another point, `x = [1.1, 1.1]`:
=#
p = [1.1, 1.1]
ν_iter = ForwardDiff.gradient(d_implicit_iter, p)
@show ν_iter, norm(ν_iter)
@test all(abs.(ν_iter .- √(2) / 2) .< 1e-9) # src
ν_direct = ForwardDiff.gradient(d_impl_direct, p)
@show ν_direct, norm(ν_direct)
@test all(abs.(ν_direct .- √(2) / 2) .< 1e-9) # src

#=
Using the `DirectLinearSolver`, we can even access the hessian of our distance field. Analytically,
it reads:
```math
\nabla(\nabla d)(x) = \dfrac{1}{\lVert x \rVert^3} \begin{pmatrix}
    x_2^ & -x_1 x_2 \\
    -x_1 x_2 & x_2^2 \\
\end{pmatrix}.
```
=#
p = [1.1, 1.2]
ℋ_iter = ForwardDiff.hessian(d_implicit_iter, p)
ℋ_ref = [(p[2]^2) (-p[1]*p[2]); (-p[1]*p[2]) (p[1]^2)] ./ norm(p)^3
@show abs.(ℋ_iter .- ℋ_ref)
@test all(abs.(ℋ_iter .- ℋ_ref) .< 1e-9) # src
