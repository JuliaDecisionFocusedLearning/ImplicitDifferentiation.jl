import AbstractDifferentiation as AD
using ChainRulesCore
using ChainRulesTestUtils
using FiniteDifferences: FiniteDifferences
using ForwardDiff: ForwardDiff
import ImplicitDifferentiation as ID
using ImplicitDifferentiation: ImplicitFunction, identity_break_autodiff
using ImplicitDifferentiation: DirectLinearSolver, IterativeLinearSolver
using JET
using LinearAlgebra
using Random
using ReverseDiff: ReverseDiff
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
    return sqrt.(identity_break_autodiff(abs.(x)))
end

function mypower(x::AbstractArray, p)
    return identity_break_autodiff(abs.(x)) .^ p
end

## Various signatures

function make_implicit_sqrt(; kwargs...)
    forward(x) = mysqrt(x)
    conditions(x, y) = y .^ 2 .- abs.(x)
    implicit = ImplicitFunction(forward, conditions; kwargs...)
    return implicit
end

function make_implicit_sqrt_byproduct(; kwargs...)
    function forward(x)
        z = 0.5
        return mypower(x, z), z
    end
    conditions(x, y, z) = y .^ (1 / z) .- x
    implicit = ImplicitFunction(forward, conditions; kwargs...)
    return implicit
end

function make_implicit_power_args(; kwargs...)
    forward(x, p) = mypower(x, p)
    conditions(x, y, p) = y .^ (1 / p) .- abs.(x)
    implicit = ImplicitFunction(forward, conditions; kwargs...)
    return implicit
end

function make_implicit_power_kwargs(; kwargs...)
    forward(x; p) = mypower(x, p)
    conditions(x, y; p) = y .^ (1 / p) .- abs.(x)
    implicit = ImplicitFunction(forward, conditions; kwargs...)
    return implicit
end

## Low level tests

function test_implicit_call(x; kwargs...)
    imf1 = make_implicit_sqrt(; kwargs...)
    imf2 = make_implicit_sqrt_byproduct(; kwargs...)
    imf3 = make_implicit_power_args(; kwargs...)
    imf4 = make_implicit_power_kwargs(; kwargs...)

    y_true = sqrt.(x)
    y1 = @inferred imf1(x)
    y2, z2 = @inferred imf2(x)
    y3 = @inferred imf3(x, 0.5)
    y4 = @inferred imf4(x; p=0.5)

    @testset "Exact value" begin
        @test y1 ≈ y_true
        @test y2 ≈ y_true
        @test y3 ≈ y_true
        @test y4 ≈ y_true
        @test z2 ≈ 0.5
    end

    if typeof(x) <: StaticArray
        @testset "Static arrays" begin
            @test is_static_array(y1)
            @test is_static_array(y2)
            @test is_static_array(y3)
            @test is_static_array(y4)
        end
    end

    @testset "JET" begin
        @test_opt target_modules = (ID,) imf1(x)
        @test_opt target_modules = (ID,) imf2(x)
        @test_opt target_modules = (ID,) imf3(x, 0.5)
        @test_opt target_modules = (ID,) imf4(x; p=0.5)

        @test_call target_modules = (ID,) imf1(x)
        @test_call target_modules = (ID,) imf2(x)
        @test_call target_modules = (ID,) imf3(x, 0.5)
        @test_call target_modules = (ID,) imf4(x; p=0.5)
    end
end

function test_implicit_duals(x; kwargs...)
    imf1 = make_implicit_sqrt(; kwargs...)
    imf2 = make_implicit_sqrt_byproduct(; kwargs...)
    imf3 = make_implicit_power_args(; kwargs...)
    imf4 = make_implicit_power_kwargs(; kwargs...)

    y_true = sqrt.(x)
    x_and_dx = ForwardDiff.Dual.(x, ((0, 1),))

    y_true = sqrt.(x)
    y_and_dy1 = @inferred imf1(x_and_dx)
    y_and_dy2, z2 = @inferred imf2(x_and_dx)
    y_and_dy3 = @inferred imf3(x_and_dx, 0.5)
    y_and_dy4 = @inferred imf4(x_and_dx; p=0.5)

    @testset "Dual numbers" begin
        @test ForwardDiff.value.(y_and_dy1) ≈ y_true
        @test ForwardDiff.value.(y_and_dy2) ≈ y_true
        @test ForwardDiff.value.(y_and_dy3) ≈ y_true
        @test ForwardDiff.value.(y_and_dy4) ≈ y_true
        @test z2 ≈ 0.5
    end

    if typeof(x) <: StaticArray
        @testset "Static arrays" begin
            @test is_static_array(y_and_dy1)
            @test is_static_array(y_and_dy2)
            @test is_static_array(y_and_dy3)
            @test is_static_array(y_and_dy4)
        end
    end

    @testset "JET" begin
        @test_opt target_modules = (ID,) imf1(x_and_dx)
        @test_opt target_modules = (ID,) imf2(x_and_dx)
        @test_opt target_modules = (ID,) imf3(x_and_dx, 0.5)
        @test_opt target_modules = (ID,) imf4(x_and_dx; p=0.5)

        @test_call target_modules = (ID,) imf1(x_and_dx)
        @test_call target_modules = (ID,) imf2(x_and_dx)
        @test_call target_modules = (ID,) imf3(x_and_dx, 0.5)
        @test_call target_modules = (ID,) imf4(x_and_dx; p=0.5)
    end
end

function test_implicit_rrule(rc, x; kwargs...)
    imf1 = make_implicit_sqrt(; kwargs...)
    imf2 = make_implicit_sqrt_byproduct(; kwargs...)
    imf3 = make_implicit_power_args(; kwargs...)
    imf4 = make_implicit_power_kwargs(; kwargs...)

    y_true = sqrt.(x)
    dy = rand(eltype(y_true), size(y_true)...)
    dz = nothing

    y1, pb1 = @inferred rrule(rc, imf1, x)
    (y2, z2), pb2 = @inferred rrule(rc, imf2, x)
    y3, pb3 = @inferred rrule(rc, imf3, x, 0.5)
    y4, pb4 = @inferred rrule(rc, imf4, x; p=0.5)

    dimf1, dx1 = @inferred pb1(dy)
    dimf2, dx2 = @inferred pb2((dy, dz))
    dimf3, dx3, dp3 = @inferred pb3(dy)
    dimf4, dx4 = @inferred pb4(dy)

    @testset "Pullbacks" begin
        @test y1 ≈ y_true
        @test y2 ≈ y_true
        @test y3 ≈ y_true
        @test y4 ≈ y_true
        @test z2 ≈ 0.5

        @test dimf1 isa NoTangent
        @test dimf2 isa NoTangent
        @test dimf3 isa NoTangent
        @test dimf4 isa NoTangent

        @test size(dx1) == size(x)
        @test size(dx2) == size(x)
        @test size(dx3) == size(x)
        @test size(dx4) == size(x)

        @test dp3 isa ChainRulesCore.NotImplemented
    end

    if typeof(x) <: StaticArray
        @testset "Static arrays" begin
            @test is_static_array(y1)
            @test is_static_array(y2)
            @test is_static_array(y3)
            @test is_static_array(y4)

            @test is_static_array(dx1)
            @test is_static_array(dx2)
            @test is_static_array(dx3)
            @test is_static_array(dx4)
        end
    end

    @testset "JET" begin
        @test_skip @test_opt target_modules = (ID,) rrule(rc, imf1, x)
        @test_skip @test_opt target_modules = (ID,) rrule(rc, imf2, x)
        @test_skip @test_opt target_modules = (ID,) rrule(rc, imf3, x, 0.5)
        @test_skip @test_opt target_modules = (ID,) rrule(rc, imf4, x; p=0.5)

        @test_skip @test_opt target_modules = (ID,) pb1(dy)
        @test_skip @test_opt target_modules = (ID,) pb2((dy, dz))
        @test_skip @test_opt target_modules = (ID,) pb3(dy)
        @test_skip @test_opt target_modules = (ID,) pb4(dy)

        @test_call target_modules = (ID,) rrule(rc, imf1, x)
        @test_call target_modules = (ID,) rrule(rc, imf2, x)
        @test_call target_modules = (ID,) rrule(rc, imf3, x, 0.5)
        @test_call target_modules = (ID,) rrule(rc, imf4, x; p=0.5)

        @test_call target_modules = (ID,) pb1(dy)
        @test_call target_modules = (ID,) pb2((dy, dz))
        @test_call target_modules = (ID,) pb3(dy)
        @test_call target_modules = (ID,) pb4(dy)
    end

    @testset "ChainRulesTestUtils" begin
        test_rrule(rc, imf1, x; atol=1e-2)
        test_rrule(rc, imf2, x; atol=1e-2)
        test_rrule(rc, imf3, x, 0.5; atol=1e-2)
        test_rrule(rc, imf4, x; atol=1e-2, fkwargs=(p=0.5,))
    end
end

## High-level tests per backend

function test_implicit_forwarddiff(x; kwargs...)
    imf1 = make_implicit_sqrt(; kwargs...)
    imf2 = make_implicit_sqrt_byproduct(; kwargs...)
    imf3 = make_implicit_power_args(; kwargs...)
    imf4 = make_implicit_power_kwargs(; kwargs...)

    J_true = Diagonal(0.5 ./ vec(sqrt.(x)))
    J1 = ForwardDiff.jacobian(imf1, x)
    J2 = ForwardDiff.jacobian(first ∘ imf2, x)
    J3 = ForwardDiff.jacobian(_x -> imf3(_x, 0.5), x)
    J4 = ForwardDiff.jacobian(_x -> imf4(_x; p=0.5), x)

    @testset "Exact Jacobian" begin
        @test J1 ≈ J_true
        @test J2 ≈ J_true
        @test J3 ≈ J_true
        @test J4 ≈ J_true
    end
    return nothing
end

function test_implicit_zygote(x; kwargs...)
    imf1 = make_implicit_sqrt(; kwargs...)
    imf2 = make_implicit_sqrt_byproduct(; kwargs...)
    imf3 = make_implicit_power_args(; kwargs...)
    imf4 = make_implicit_power_kwargs(; kwargs...)

    J_true = Diagonal(0.5 ./ vec(sqrt.(x)))
    J1 = Zygote.jacobian(imf1, x)[1]
    J2 = Zygote.jacobian(first ∘ imf2, x)[1]
    J3 = Zygote.jacobian(imf3, x, 0.5)[1]
    J4 = Zygote.jacobian(_x -> imf4(_x; p=0.5), x)[1]

    @testset "Exact Jacobian" begin
        @test J1 ≈ J_true
        @test J2 ≈ J_true
        @test J3 ≈ J_true
        @test J4 ≈ J_true
    end
    return nothing
end

function test_implicit(x; kwargs...)
    @testset verbose = true "Call" begin
        test_implicit_call(x; kwargs...)
    end
    @testset verbose = true "ForwardDiff.jl" begin
        test_implicit_forwarddiff(x; kwargs...)
        test_implicit_duals(x; kwargs...)
    end
    @testset verbose = true "Zygote.jl" begin
        rc = Zygote.ZygoteRuleConfig()
        test_implicit_zygote(x; kwargs...)
        test_implicit_rrule(rc, x; kwargs...)
    end
    return nothing
end

## Parameter combinations

linear_solver_candidates = (
    IterativeLinearSolver(), #
    DirectLinearSolver(), #
)

conditions_backend_candidates = (
    nothing,  #
    AD.ForwardDiffBackend(),  #
    # AD.ZygoteBackend(),  # TODO: failing
    # AD.ReverseDiffBackend()  # TODO: failing
    # AD.FiniteDifferencesBackend()  # TODO: failing
);

x_candidates = (
    rand(2, 3, 4), #
    SArray{Tuple{2,3,4}}(rand(2, 3, 4)), #
);

params_candidates = []

for linear_solver in linear_solver_candidates, x in x_candidates
    push!(
        params_candidates, (;
            linear_solver=linear_solver, #
            conditions_backend=nothing, #
            x=x, #
        )
    )
end

for conditions_backend in conditions_backend_candidates
    push!(
        params_candidates,
        (;
            linear_solver=linear_solver_candidates[1], #
            conditions_backend=conditions_backend, #
            x=x_candidates[1], #
        ),
    )
end

## Test loop

for (linear_solver, conditions_backend, x) in params_candidates
    x isa StaticArray && linear_solver isa IterativeLinearSolver && continue
    testsetname = "$(typeof(linear_solver)) - $(typeof(conditions_backend)) - $(typeof(x))"
    @info "$testsetname"
    @testset "$testsetname" begin
        test_implicit(x; linear_solver, conditions_backend)
    end
end
