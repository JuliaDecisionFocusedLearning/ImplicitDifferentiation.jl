import AbstractDifferentiation as AD
using ChainRulesCore
using ChainRulesTestUtils
using ForwardDiff: ForwardDiff
import ImplicitDifferentiation as ID
using ImplicitDifferentiation: ImplicitFunction, HandleByproduct, ReturnByproduct
using ImplicitDifferentiation: DirectLinearSolver, IterativeLinearSolver
using ImplicitDifferentiation: identity_break_autodiff, handles_byproduct
using JET
using LinearAlgebra
using Random
using StaticArrays
using Test
using Zygote: Zygote, ZygoteRuleConfig

@static if VERSION < v"1.9"
    macro test_opt(x...)
        return :()
    end
    macro test_call(x...)
        return :()
    end
end

Random.seed!(63);

## Utils

function is_static_array(a)
    return (
        typeof(a) <: StaticArray ||
        typeof(a) <: (Base.ReshapedArray{T,N,<:StaticArray} where {T,N})
    )
end

function mysqrt(x::AbstractArray)
    return sqrt.(identity_break_autodiff(x))
end

function mysqrt_byproduct(x::AbstractArray)
    z = rand((2,))
    y = identity_break_autodiff(x) .^ (1 / z)
    return y, z
end

function make_implicit_sqrt(; kwargs...)
    forward(x) = mysqrt(x)
    conditions(x, y) = y .^ 2 .- x
    implicit = ImplicitFunction(forward, conditions; kwargs...)
    return implicit
end

function make_implicit_sqrt_byproduct(; kwargs...)
    forward(x) = mysqrt_byproduct(x)
    conditions(x, y, z) = y .^ z .- x
    implicit = ImplicitFunction(forward, conditions, HandleByproduct(); kwargs...)
    return implicit
end

## Low level tests

function test_implicit_call(implicit, x; y_true)
    y1 = @inferred implicit(x)
    y2, z2 = @inferred implicit(x, ReturnByproduct())
    @testset "Exact value" begin
        @test y1 ≈ y_true
        @test y2 ≈ y_true
    end
    @testset "Byproduct" begin
        if handles_byproduct(implicit)
            @test z2 == 2
        else
            @test z2 == 0
        end
    end
    if typeof(x) <: StaticArray
        @testset "Static arrays" begin
            @test is_static_array(y1)
            @test is_static_array(y2)
        end
    end
    @testset "JET" begin
        @test_opt target_modules = (ID,) implicit(x)
        @test_call target_modules = (ID,) implicit(x)
    end
end

function test_implicit_duals(implicit, x; y_true)
    x_and_dx = ForwardDiff.Dual.(x, ((0, 0),))
    y_and_dy1 = @inferred implicit(x_and_dx)
    y_and_dy2, z2 = @inferred implicit(x_and_dx, ReturnByproduct())
    @testset "Dual numbers" begin
        @test size(y_and_dy1) == size(y_true)
        @test size(y_and_dy2) == size(y_true)
        @test ForwardDiff.value.(y_and_dy1) ≈ y_true
        @test ForwardDiff.value.(y_and_dy2) ≈ y_true
    end
    @testset "Byproduct" begin
        if handles_byproduct(implicit)
            @test z2 == 2
        else
            @test z2 == 0
        end
    end
    if typeof(x) <: StaticArray
        @testset "Static arrays" begin
            @test is_static_array(y_and_dy1)
            @test is_static_array(y_and_dy2)
        end
    end
    @testset "JET" begin
        @test_opt target_modules = (ID,) implicit(x_and_dx)
        @test_call target_modules = (ID,) implicit(x_and_dx)
    end
end

function test_implicit_rrule(rc, implicit, x; y_true, J_true)
    y1, pb1 = @inferred rrule(rc, implicit, x)
    (y2, z2), pb2 = @inferred rrule(rc, implicit, x, ReturnByproduct())
    dy1 = rand(eltype(y1), size(y1)...)
    dy2 = rand(eltype(y2), size(y2)...)
    dz2 = nothing
    dimp1, dx1 = @inferred pb1(dy1)
    dimp2, dx2, drp = @inferred pb2((dy2, dz2))
    @testset "Pullbacks" begin
        @test y1 ≈ y_true
        @test y2 ≈ y_true
        @test size(dx1) == size(x)
        @test size(dx2) == size(x)
        @test dimp1 isa NoTangent
        @test dimp2 isa NoTangent
        @test drp isa NoTangent
    end
    @testset "Byproduct" begin
        if handles_byproduct(implicit)
            @test z2 == 2
        else
            @test z2 == 0
        end
    end
    if typeof(x) <: StaticArray
        @testset "Static arrays" begin
            @test is_static_array(y1)
            @test is_static_array(y2)
            @test is_static_array(dx1)
            @test is_static_array(dx2)
        end
    end
    @testset "JET" begin
        @test_skip @test_opt target_modules = (ID,) rrule(rc, implicit, x)
        @test_skip @test_opt target_modules = (ID,) pb1(dy1)
        @test_call target_modules = (ID,) rrule(rc, implicit, x)
        @test_call target_modules = (ID,) pb1(dy1)
    end
    @testset "ChainRulesTestUtils" begin
        # Skipped because of https://github.com/JuliaDiff/ChainRulesTestUtils.jl/issues/232 and because it detects weird type instabilities
        @test_skip test_rrule(implicit, x)
        @test_skip test_rrule(implicit, x, ReturnByproduct())
    end
end

## High-level tests per backend

function test_implicit_forwarddiff(implicit, x; y_true, J_true)
    J1 = ForwardDiff.jacobian(implicit, x)
    J2 = ForwardDiff.jacobian(x -> implicit(x, ReturnByproduct())[1], x)
    @testset "Exact Jacobian" begin
        @test J1 ≈ J_true
        @test J2 ≈ J_true
    end
    test_implicit_duals(implicit, x; y_true)
    return nothing
end

function test_implicit_zygote(implicit, x; y_true, J_true)
    J1 = Zygote.jacobian(implicit, x)[1]
    J2 = Zygote.jacobian(x -> implicit(x, ReturnByproduct())[1], x)[1]
    @testset "Exact Jacobian" begin
        @test J1 ≈ J_true
        @test J2 ≈ J_true
    end
    rc = Zygote.ZygoteRuleConfig()
    test_implicit_rrule(rc, implicit, x; y_true, J_true)
    return nothing
end

function test_implicit(implicit, x; y_true, J_true)
    @testset "Call" begin
        test_implicit_call(implicit, x; y_true)
    end
    @testset "ForwardDiff.jl" begin
        test_implicit_forwarddiff(implicit, x; y_true, J_true)
    end
    @testset "Zygote.jl" begin
        test_implicit_zygote(implicit, x; y_true, J_true)
    end
    return nothing
end

## Actual loop

x_candidates = (
    rand(2), rand(2, 3, 4), SVector{2}(rand(2)), SArray{Tuple{2,3,4}}(rand(2, 3, 4))
);

linear_solver_candidates = (IterativeLinearSolver(), DirectLinearSolver())

for linear_solver in linear_solver_candidates
    implicit_variants = (
        make_implicit_sqrt(; linear_solver), make_implicit_sqrt_byproduct(; linear_solver)
    )
    for implicit in implicit_variants, x in x_candidates
        x isa StaticArray && linear_solver isa IterativeLinearSolver && continue

        y_true = sqrt.(x)
        J_true = Diagonal(0.5 ./ vec(sqrt.(x)))

        testsetname = "$(typeof(linear_solver)) - $(handles_byproduct(implicit)) - $(typeof(x))"

        @info "Systematic tests - $testsetname"
        @testset "$testsetname" begin
            test_implicit(implicit, x; y_true, J_true)
        end
    end
end
