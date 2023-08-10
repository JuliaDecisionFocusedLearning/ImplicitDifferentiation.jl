import AbstractDifferentiation as AD
using ChainRulesCore
using ChainRulesTestUtils
using ForwardDiff: ForwardDiff
import ImplicitDifferentiation as ID
using ImplicitDifferentiation: ImplicitFunction, identity_break_autodiff
using ImplicitDifferentiation: DirectLinearSolver, IterativeLinearSolver
using JET
using LinearAlgebra
using Random
using SparseArrays
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

change_shape(x::AbstractArray{T,3}) where {T} = x[:, :, 1]

function mysqrt(x::AbstractArray)
    return identity_break_autodiff(sqrt.(abs.(change_shape(x))))
end

function mypower(x::AbstractArray, p)
    return identity_break_autodiff(abs.(change_shape(x)) .^ p)
end

## Various signatures

function make_implicit_sqrt(; kwargs...)
    forward(x) = mysqrt(x)
    conditions(x, y) = y .^ 2 .- abs.(change_shape(x))
    implicit = ImplicitFunction(forward, conditions; kwargs...)
    return implicit
end

function make_implicit_sqrt_byproduct(; kwargs...)
    forward(x) = mysqrt(x), eltype(x)(2)  # fails rrule tests with 2 integer, probably due to https://juliadiff.org/ChainRulesTestUtils.jl/dev/index.html#Specifying-Tangents
    conditions(x, y, z::Number) = y .^ z .- abs.(change_shape(x))
    implicit = ImplicitFunction(forward, conditions; kwargs...)
    return implicit
end

function make_implicit_power_args(; kwargs...)
    forward(x, p::Number) = mypower(x, one(eltype(x)) / p)
    conditions(x, y, p::Number) = y .^ p .- abs.(change_shape(x))
    implicit = ImplicitFunction(forward, conditions; kwargs...)
    return implicit
end

function make_implicit_power_kwargs(; kwargs...)
    forward(x; p::Number) = mypower(x, one(eltype(x)) / p)
    conditions(x, y; p::Number) = y .^ p .- abs.(change_shape(x))
    implicit = ImplicitFunction(forward, conditions; kwargs...)
    return implicit
end

## Low level tests

function coherent_array_type(a, b)
    if a isa Array
        return b isa Array || b isa (Base.ReshapedArray{T,N,<:Array} where {T,N})
    elseif a isa StaticArray
        return b isa StaticArray ||
               b isa (Base.ReshapedArray{T,N,<:StaticArray} where {T,N})
    elseif a isa AbstractSparseArray
        return b isa AbstractSparseArray ||
               b isa (Base.ReshapedArray{T,N,<:AbstractSparseArray} where {T,N})
    else
        error("New array type")
    end
end

function test_implicit_call(x::AbstractArray{T}; kwargs...) where {T}
    imf1 = make_implicit_sqrt(; kwargs...)
    imf2 = make_implicit_sqrt_byproduct(; kwargs...)
    imf3 = make_implicit_power_args(; kwargs...)
    imf4 = make_implicit_power_kwargs(; kwargs...)

    y_true = mysqrt(x)
    y1 = @inferred imf1(x)
    y2, z2 = @inferred imf2(x)
    y3 = @inferred imf3(x, 2)
    y4 = @inferred imf4(x; p=2)

    @testset "Exact value" begin
        @test y1 ≈ y_true
        @test y2 ≈ y_true
        @test y3 ≈ y_true
        @test y4 ≈ y_true
        @test z2 ≈ 2
    end

    @testset "Array type" begin
        @test coherent_array_type(x, y1)
        @test coherent_array_type(x, y2)
        @test coherent_array_type(x, y3)
        @test coherent_array_type(x, y4)
    end

    @testset "JET" begin
        @test_opt target_modules = (ID,) imf1(x)
        @test_opt target_modules = (ID,) imf2(x)
        @test_opt target_modules = (ID,) imf3(x, 2)
        @test_opt target_modules = (ID,) imf4(x; p=2)

        @test_call target_modules = (ID,) imf1(x)
        @test_call target_modules = (ID,) imf2(x)
        @test_call target_modules = (ID,) imf3(x, 2)
        @test_call target_modules = (ID,) imf4(x; p=2)
    end
end

function test_implicit_duals(x::AbstractArray{T}; kwargs...) where {T}
    imf1 = make_implicit_sqrt(; kwargs...)
    imf2 = make_implicit_sqrt_byproduct(; kwargs...)
    imf3 = make_implicit_power_args(; kwargs...)
    imf4 = make_implicit_power_kwargs(; kwargs...)

    y_true = mysqrt(x)
    dx = similar(x)
    dx .= one(T)
    x_and_dx = ForwardDiff.Dual.(x, dx)

    y_and_dy1 = @inferred imf1(x_and_dx)
    y_and_dy2, z2 = @inferred imf2(x_and_dx)
    y_and_dy3 = @inferred imf3(x_and_dx, 2)
    y_and_dy4 = @inferred imf4(x_and_dx; p=2)

    @testset "Dual numbers" begin
        @test ForwardDiff.value.(y_and_dy1) ≈ y_true
        @test ForwardDiff.value.(y_and_dy2) ≈ y_true
        @test ForwardDiff.value.(y_and_dy3) ≈ y_true
        @test ForwardDiff.value.(y_and_dy4) ≈ y_true
        @test z2 ≈ 2
    end

    @testset "Static arrays" begin
        @test coherent_array_type(x, y_and_dy1)
        @test coherent_array_type(x, y_and_dy2)
        @test coherent_array_type(x, y_and_dy3)
        @test coherent_array_type(x, y_and_dy4)
    end

    @testset "JET" begin
        @test_opt target_modules = (ID,) imf1(x_and_dx)
        @test_opt target_modules = (ID,) imf2(x_and_dx)
        @test_opt target_modules = (ID,) imf3(x_and_dx, 2)
        @test_opt target_modules = (ID,) imf4(x_and_dx; p=2)

        @test_call target_modules = (ID,) imf1(x_and_dx)
        @test_call target_modules = (ID,) imf2(x_and_dx)
        @test_call target_modules = (ID,) imf3(x_and_dx, 2)
        @test_call target_modules = (ID,) imf4(x_and_dx; p=2)
    end
end

function test_implicit_rrule(rc, x::AbstractArray{T}; kwargs...) where {T}
    imf1 = make_implicit_sqrt(; kwargs...)
    imf2 = make_implicit_sqrt_byproduct(; kwargs...)
    imf3 = make_implicit_power_args(; kwargs...)
    imf4 = make_implicit_power_kwargs(; kwargs...)

    y_true = mysqrt(x)
    dy = similar(y_true)
    dy .= one(eltype(y_true))
    dz = nothing

    y1, pb1 = @inferred rrule(rc, imf1, x)
    (y2, z2), pb2 = @inferred rrule(rc, imf2, x)
    y3, pb3 = @inferred rrule(rc, imf3, x, 2)
    y4, pb4 = @inferred rrule(rc, imf4, x; p=2)

    dimf1, dx1 = @inferred pb1(dy)
    dimf2, dx2 = @inferred pb2((dy, dz))
    dimf3, dx3, dp3 = @inferred pb3(dy)
    dimf4, dx4 = @inferred pb4(dy)

    @testset "Pullbacks" begin
        @test y1 ≈ y_true
        @test y2 ≈ y_true
        @test y3 ≈ y_true
        @test y4 ≈ y_true
        @test z2 ≈ 2

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

    @testset "Array type" begin
        @test coherent_array_type(x, y1)
        @test coherent_array_type(x, y2)
        @test coherent_array_type(x, y3)
        @test coherent_array_type(x, y4)

        @test coherent_array_type(x, dx1)
        @test coherent_array_type(x, dx2)
        @test coherent_array_type(x, dx3)
        @test coherent_array_type(x, dx4)
    end

    @testset "JET" begin
        @test_skip @test_opt target_modules = (ID,) rrule(rc, imf1, x)
        @test_skip @test_opt target_modules = (ID,) rrule(rc, imf2, x)
        @test_skip @test_opt target_modules = (ID,) rrule(rc, imf3, x, 2)
        @test_skip @test_opt target_modules = (ID,) rrule(rc, imf4, x; p=2)

        @test_skip @test_opt target_modules = (ID,) pb1(dy)
        @test_skip @test_opt target_modules = (ID,) pb2((dy, dz))
        @test_skip @test_opt target_modules = (ID,) pb3(dy)
        @test_skip @test_opt target_modules = (ID,) pb4(dy)

        @test_call target_modules = (ID,) rrule(rc, imf1, x)
        @test_call target_modules = (ID,) rrule(rc, imf2, x)
        @test_call target_modules = (ID,) rrule(rc, imf3, x, 2)
        @test_call target_modules = (ID,) rrule(rc, imf4, x; p=2)

        @test_call target_modules = (ID,) pb1(dy)
        @test_call target_modules = (ID,) pb2((dy, dz))
        @test_call target_modules = (ID,) pb3(dy)
        @test_call target_modules = (ID,) pb4(dy)
    end

    @testset "ChainRulesTestUtils" begin
        test_rrule(rc, imf1, x; atol=1e-2)
        test_rrule(rc, imf2, x; atol=5e-2)
        test_rrule(rc, imf3, x, 2; atol=1e-2)
        test_rrule(rc, imf4, x; atol=1e-2, fkwargs=(p=2,))
    end
end

## High-level tests per backend

function test_implicit_forwarddiff(x::AbstractArray{T}; kwargs...) where {T}
    imf1 = make_implicit_sqrt(; kwargs...)
    imf2 = make_implicit_sqrt_byproduct(; kwargs...)
    imf3 = make_implicit_power_args(; kwargs...)
    imf4 = make_implicit_power_kwargs(; kwargs...)

    J1 = ForwardDiff.jacobian(imf1, x)
    J2 = ForwardDiff.jacobian(first ∘ imf2, x)
    J3 = ForwardDiff.jacobian(_x -> imf3(_x, 2), x)
    J4 = ForwardDiff.jacobian(_x -> imf4(_x; p=2), x)
    J_true = ForwardDiff.jacobian(_x -> sqrt.(change_shape(_x)), x)

    @testset "Exact Jacobian" begin
        @test J1 ≈ J_true
        @test J2 ≈ J_true
        @test J3 ≈ J_true
        @test J4 ≈ J_true

        @test eltype(J1) == eltype(x)
        @test eltype(J2) == eltype(x)
        @test eltype(J3) == eltype(x)
        @test eltype(J4) == eltype(x)
    end
    return nothing
end

function test_implicit_zygote(x::AbstractArray{T}; kwargs...) where {T}
    imf1 = make_implicit_sqrt(; kwargs...)
    imf2 = make_implicit_sqrt_byproduct(; kwargs...)
    imf3 = make_implicit_power_args(; kwargs...)
    imf4 = make_implicit_power_kwargs(; kwargs...)

    J1 = Zygote.jacobian(imf1, x)[1]
    J2 = Zygote.jacobian(first ∘ imf2, x)[1]
    J3 = Zygote.jacobian(imf3, x, 2)[1]
    J4 = Zygote.jacobian(_x -> imf4(_x; p=2), x)[1]
    J_true = Zygote.jacobian(_x -> sqrt.(change_shape(_x)), x)[1]

    @testset "Exact Jacobian" begin
        @test J1 ≈ J_true
        @test J2 ≈ J_true
        @test J3 ≈ J_true
        @test J4 ≈ J_true

        @test eltype(J1) == eltype(x)
        @test eltype(J2) == eltype(x)
        @test eltype(J3) == eltype(x)
        @test eltype(J4) == eltype(x)
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
    rand(Float32, 2, 3, 2), #
    SArray{Tuple{2,3,2}}(rand(Float32, 2, 3, 2)), #
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
    testsetname = "$(typeof(linear_solver)) - $(typeof(conditions_backend)) - $(typeof(x))"
    @info "$testsetname"
    @testset "$testsetname" begin
        test_implicit(x; linear_solver, conditions_backend)
    end
end
