using ChainRulesCore
using ChainRulesTestUtils
using ForwardDiff
using ImplicitDifferentiation
using JET
using LinearAlgebra
using Random
using StaticArrays
using Test
using Zygote

Random.seed!(63);

"""
    mysqrt(x)

Compute the elementwise square root, breaking Zygote.jl and ForwardDiff.jl in the process.
"""
function mysqrt(x::AbstractArray)
    a = [0.0]
    a[1] = first(x)
    return sqrt.(x)
end

function make_implicit_sqrt(linear_solver)
    forward(x) = mysqrt(x)
    conditions(x, y) = y .^ 2 .- x
    implicit = ImplicitFunction(forward, conditions, linear_solver)
    return implicit
end

function test_implicit_sqrt_call(implicit, x)
    @test (@inferred implicit(x)) ≈ sqrt.(x)
    if typeof(x) <: StaticArray
        @test typeof(implicit(x)) <: StaticArray
    end
    if VERSION >= v"1.9"
        test_opt(implicit, (typeof(x),))
    end
end

function test_implicit_sqrt_forward(implicit, x)
    y = implicit(x)
    J = Diagonal(0.5 ./ vec(sqrt.(x)))
    @test ForwardDiff.jacobian(implicit, x) ≈ J
    x_and_dx = ForwardDiff.Dual.(x, ((0, 0),))
    y_and_dy = @inferred implicit(x_and_dx)
    @test size(y_and_dy) == size(y)
    if typeof(x) <: StaticArray
        @test typeof(y_and_dy) <: StaticArray ||
            typeof(y_and_dy) <: (Base.ReshapedArray{T,N,<:StaticArray} where {T,N})
    end
end

function test_implicit_sqrt_reverse(implicit, x)
    J = Diagonal(0.5 ./ vec(sqrt.(x)))
    @test Zygote.jacobian(implicit, x)[1] ≈ J
    _, pullback = @inferred rrule(Zygote.ZygoteRuleConfig(), implicit, x)
    dy = similar(implicit(x))
    dy .= zero(eltype(dy))
    @test (@inferred pullback(dy)) == pullback(dy)
    _, dx = pullback(dy)
    @test size(dx) == size(x)
    if typeof(x) <: StaticArray
        @test typeof(dx) <: StaticArray ||
            typeof(dx) <: (Base.ReshapedArray{T,N,<:StaticArray} where {T,N})
    end
    # Skipped because of https://github.com/JuliaDiff/ChainRulesTestUtils.jl/issues/232 and because it detects weird type instabilities
    @test_skip test_rrule(implicit, x)
end

x_candidates = (
    rand(2), rand(2, 3, 4), SVector{2}(rand(2)), SArray{Tuple{2,3,4}}(rand(2, 3, 4))
)

for linear_solver in (IterativeLinearSolver(), DirectLinearSolver())
    forward(x) = mysqrt(x)
    conditions(x, y) = y .^ 2 .- x
    implicit = ImplicitFunction(forward, conditions, linear_solver)
    for x in x_candidates
        testsetname = "$(typeof(x)) - $(typeof(linear_solver))"
        @testset verbose = true "$testsetname" begin
            if x isa StaticArray && linear_solver isa IterativeLinearSolver
                continue
            end
            test_implicit_sqrt_call(implicit, x)
            test_implicit_sqrt_forward(implicit, x)
            test_implicit_sqrt_reverse(implicit, x)
        end
    end
end