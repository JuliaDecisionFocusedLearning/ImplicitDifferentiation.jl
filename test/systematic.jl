using ChainRulesCore
using ChainRulesTestUtils
using ForwardDiff
using ImplicitDifferentiation
using ImplicitDifferentiation: handles_byproduct
using JET
using LinearAlgebra
using Random
using StaticArrays
using Test
using Zygote
using Zygote: ZygoteRuleConfig

@static if VERSION < v"1.9"
    macro test_opt(x...)
        return :()
    end
    macro test_call(x...)
        return :()
    end
end

Random.seed!(63);

function is_static_array(a)
    return (
        typeof(a) <: StaticArray ||
        typeof(a) <: (Base.ReshapedArray{T,N,<:StaticArray} where {T,N})
    )
end

function break_forwarddiff_zygote(x)
    a = [0.0]
    a[1] = float(first(x))
    return nothing
end

function mysqrt(x::AbstractArray)
    break_forwarddiff_zygote(x)
    return sqrt.(x)
end

function mysqrt_byproduct(x::AbstractArray)
    break_forwarddiff_zygote(x)
    z = rand((2,))
    y = x .^ (1 / z)
    return y, z
end

function make_implicit_sqrt(linear_solver)
    forward(x) = mysqrt(x)
    conditions(x, y) = y .^ 2 .- x
    implicit = ImplicitFunction(forward, conditions, linear_solver)
    return implicit
end

function make_implicit_sqrt_byproduct(linear_solver)
    forward(x) = mysqrt_byproduct(x)
    conditions(x, y, z) = y .^ z .- x
    implicit = ImplicitFunction(forward, conditions, linear_solver, HandleByproduct())
    return implicit
end

function test_implicit_call(implicit, x; y_true)
    @test_throws MethodError implicit("hello")
    @test_throws MethodError implicit(x, x)
    y1 = @inferred implicit(x)
    y2, z2 = @inferred implicit(x, ReturnByproduct())
    @test y1 ≈ y_true
    @test y2 ≈ y_true
    if typeof(x) <: StaticArray
        @test is_static_array(y1)
        @test is_static_array(y2)
    end
    if handles_byproduct(implicit)
        @test z2 == 2
    else
        @test z2 == 0
    end
    @test_opt target_modules = (ImplicitDifferentiation,) implicit(x)
    @test_call target_modules = (ImplicitDifferentiation,) implicit(x)
end

function test_implicit_forward(implicit, x; y_true, J_true)
    # High-level
    J1 = ForwardDiff.jacobian(implicit, x)
    J2 = ForwardDiff.jacobian(x -> implicit(x, ReturnByproduct())[1], x)
    @test J1 ≈ J_true
    @test J2 ≈ J_true
    # Low-level
    x_and_dx = ForwardDiff.Dual.(x, ((0, 0),))
    y_and_dy1 = @inferred implicit(x_and_dx)
    y_and_dy2, z2 = @inferred implicit(x_and_dx, ReturnByproduct())
    @test size(y_and_dy1) == size(y_true)
    @test size(y_and_dy2) == size(y_true)
    @test ForwardDiff.value.(y_and_dy1) ≈ y_true
    @test ForwardDiff.value.(y_and_dy2) ≈ y_true
    if typeof(x) <: StaticArray
        @test is_static_array(y_and_dy1)
        @test is_static_array(y_and_dy2)
    end
    if handles_byproduct(implicit)
        @test z2 == 2
    else
        @test z2 == 0
    end
    @test_opt target_modules = (ImplicitDifferentiation,) implicit(x_and_dx)
    @test_call target_modules = (ImplicitDifferentiation,) implicit(x_and_dx)
end

function test_implicit_reverse(implicit, x; y_true, J_true)
    # High-level
    J1 = Zygote.jacobian(implicit, x)[1]
    J2 = Zygote.jacobian(x -> implicit(x, ReturnByproduct())[1], x)[1]
    @test J1 ≈ J_true
    @test J2 ≈ J_true
    # Low-level
    y1, pb1 = @inferred rrule(ZygoteRuleConfig(), implicit, x)
    (y2, z2), pb2 = @inferred rrule(ZygoteRuleConfig(), implicit, x, ReturnByproduct())
    @test y1 ≈ y_true
    @test y2 ≈ y_true
    dy1 = zeros(eltype(y1), size(y1)...)
    dy2 = zeros(eltype(y2), size(y2)...)
    dz2 = nothing
    dimp1, dx1 = @inferred pb1(dy1)
    dimp2, dx2, drp = @inferred pb2((dy2, dz2))
    @test size(dx1) == size(x)
    @test size(dx2) == size(x)
    if typeof(x) <: StaticArray
        @test is_static_array(y1)
        @test is_static_array(y2)
        @test is_static_array(dx1)
        @test is_static_array(dx2)
    end
    @test dimp1 isa NoTangent
    @test dimp2 isa NoTangent
    @test drp isa NoTangent
    if handles_byproduct(implicit)
        @test z2 == 2
    else
        @test z2 == 0
    end
    @test_skip @test_opt target_modules = (ImplicitDifferentiation,) rrule(
        ZygoteRuleConfig(), implicit, x
    )
    @test_skip @test_opt target_modules = (ImplicitDifferentiation,) pb1(dy1)
    @test_call target_modules = (ImplicitDifferentiation,) rrule(
        ZygoteRuleConfig(), implicit, x
    )
    @test_call target_modules = (ImplicitDifferentiation,) pb1(dy1)
    # Skipped because of https://github.com/JuliaDiff/ChainRulesTestUtils.jl/issues/232 and because it detects weird type instabilities
    @test_skip test_rrule(implicit, x)
    @test_skip test_rrule(x -> implicit(x, ReturnByproduct()), x)
end

x_candidates = (
    rand(2), rand(2, 3, 4), SVector{2}(rand(2)), SArray{Tuple{2,3,4}}(rand(2, 3, 4))
);

linear_solver_candidates = (IterativeLinearSolver(), DirectLinearSolver())

for linear_solver in linear_solver_candidates, x in x_candidates
    if x isa StaticArray && linear_solver isa IterativeLinearSolver
        continue
    end
    y_true = sqrt.(x)
    J_true = Diagonal(0.5 ./ vec(sqrt.(x)))

    testsetname = "$(typeof(x)) - $(typeof(linear_solver))"
    implicit_sqrt = make_implicit_sqrt(linear_solver)
    implicit_sqrt_byproduct = make_implicit_sqrt_byproduct(linear_solver)

    @testset verbose = true "$testsetname" begin
        @testset "Call" begin
            test_implicit_call(implicit_sqrt, x; y_true)
            test_implicit_call(implicit_sqrt_byproduct, x; y_true)
        end
        @testset "Forward" begin
            test_implicit_forward(implicit_sqrt, x; y_true, J_true)
            test_implicit_forward(implicit_sqrt_byproduct, x; y_true, J_true)
        end
        @testset "Reverse" begin
            test_implicit_reverse(implicit_sqrt, x; y_true, J_true)
            test_implicit_reverse(implicit_sqrt_byproduct, x; y_true, J_true)
        end
    end
end
