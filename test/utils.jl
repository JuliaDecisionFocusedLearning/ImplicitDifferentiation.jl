using ADTypes
using ChainRulesCore
using ChainRulesTestUtils
using DifferentiationInterface: DifferentiationInterface
using ForwardDiff: ForwardDiff
import ImplicitDifferentiation as ID
using ImplicitDifferentiation: ImplicitFunction
using JET
using LinearAlgebra
using Test
using Zygote: Zygote, ZygoteRuleConfig

##

function identity_break_autodiff(x::X)::X where {R,X<:AbstractVector{R}}
    float(first(x))  # break ForwardDiff
    (Vector{R}(undef, 1))[1] = first(x)  # break Zygote
    result = try
        throw(copy(x))
    catch y
        y
    end
    return result
end

mysqrt(x::AbstractVector) = identity_break_autodiff(sqrt.(x))

## Various signatures

function make_implicit_sqrt_byproduct(x; kwargs...)
    forward(x) = 1 .* vcat(mysqrt(x), -mysqrt(x)), 1
    conditions(x, y, z) = abs2.(y ./ z) .- vcat(x, x)
    input_example = (copy(x),)
    implicit = ImplicitFunction(forward, conditions; input_example, kwargs...)
    return implicit
end

function make_implicit_sqrt_args(x; kwargs...)
    forward(x, p) = p .* vcat(mysqrt(x), -mysqrt(x)), nothing
    conditions(x, y, z, p) = abs2.(y ./ p) .- vcat(x, x)
    input_example = (copy(x), 2)
    implicit = ImplicitFunction(forward, conditions; input_example, kwargs...)
    return implicit
end

function test_implicit_call(x::AbstractVector{T}; kwargs...) where {T}
    imf1 = make_implicit_sqrt_byproduct(x; kwargs...)
    imf2 = make_implicit_sqrt_args(x; kwargs...)

    y_true = vcat(mysqrt(x), -mysqrt(x))
    y1, z1 = imf1(x)
    y2, z2 = imf2(x, 3)

    @testset "Primal value" begin
        @test y1 ≈ y_true
        @test y2 ≈ 3y_true
        @test z1 == 1
        @test z2 === nothing
    end
end

tag(::AbstractVector{<:ForwardDiff.Dual{T}}) where {T} = T

function test_implicit_duals(x::AbstractVector{T}; kwargs...) where {T}
    imf1 = make_implicit_sqrt_byproduct(x; kwargs...)
    imf2 = make_implicit_sqrt_args(x; kwargs...)

    y_true = vcat(mysqrt(x), -mysqrt(x))
    dx = similar(x)
    dx .= 2 * one(T)
    x_and_dx = ForwardDiff.Dual.(x, dx)

    y_and_dy1, z1 = imf1(x_and_dx)
    y_and_dy2, z2 = imf2(x_and_dx, 3)

    @testset "Dual numbers" begin
        @test ForwardDiff.value.(y_and_dy1) ≈ y_true
        @test ForwardDiff.value.(y_and_dy2) ≈ 3y_true
        @test ForwardDiff.extract_derivative(tag(y_and_dy1), y_and_dy1) ≈
            2 .* inv.(2 .* vcat(sqrt.(x), -sqrt.(x)))
        @test ForwardDiff.extract_derivative(tag(y_and_dy2), y_and_dy2) ≈
            3 .* 2 .* inv.(2 .* vcat(sqrt.(x), -sqrt.(x)))
        @test z1 == 1
        @test z2 === nothing
    end
end

function test_implicit_rrule(rc, x::AbstractVector{T}; kwargs...) where {T}
    imf1 = make_implicit_sqrt_byproduct(x; kwargs...)
    imf2 = make_implicit_sqrt_args(x; kwargs...)

    y_true = vcat(mysqrt(x), -mysqrt(x))
    dy = zero(y_true)
    dy[1:(end ÷ 2)] .= one(eltype(y_true))
    dz = nothing

    (y1, z1), pb1 = rrule(rc, imf1, x)
    (y2, z2), pb2 = rrule(rc, imf2, x, 3)

    dimf1, dx1 = pb1((dy, dz))
    dimf2, dx2, dp2 = pb2((dy, dz))

    @testset "Pullbacks" begin
        @test y1 ≈ y_true
        @test y2 ≈ 3y_true
        @test z1 == 1
        @test z2 === nothing

        @test dimf1 isa NoTangent
        @test dimf2 isa NoTangent

        @test dx2 ≈ 3 .* dx1
        @test dp2 isa ChainRulesCore.NotImplemented
    end
end

## High-level tests per backend

function test_implicit_backend(
    outer_backend::ADTypes.AbstractADType, x::AbstractVector{T}; kwargs...
) where {T}
    imf1 = make_implicit_sqrt_byproduct(x; kwargs...)
    imf2 = make_implicit_sqrt_args(x; kwargs...)

    J1 = DifferentiationInterface.jacobian(first ∘ imf1, outer_backend, x)
    J2 = DifferentiationInterface.jacobian(_x -> (first ∘ imf2)(_x, 3), outer_backend, x)

    J_true = ForwardDiff.jacobian(_x -> vcat(sqrt.(_x), -sqrt.(_x)), x)

    @testset "Exact Jacobian" begin
        @test J1 ≈ J_true
        @test J2 ≈ 3 .* J_true
    end
    return nothing
end

function test_implicit(outer_backends, x; kwargs...)
    @testset "Call" begin
        test_implicit_call(x; kwargs...)
    end
    @testset "Duals" begin
        test_implicit_duals(x; kwargs...)
    end
    @testset "ChainRule" begin
        test_implicit_rrule(ZygoteRuleConfig(), x; kwargs...)
    end
    @testset "Jacobian - $outer_backend" for outer_backend in outer_backends
        test_implicit_backend(outer_backend, x; kwargs...)
    end
    return nothing
end
