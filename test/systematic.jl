using ADTypes
using ChainRulesCore
using ChainRulesTestUtils
using DifferentiationInterface: DifferentiationInterface
using Enzyme: Enzyme
using ForwardDiff: ForwardDiff
import ImplicitDifferentiation as ID
using ImplicitDifferentiation: ImplicitFunction
using JET
using Krylov
using LinearAlgebra
using Random
using SparseArrays
using StaticArrays
using Test
using Zygote: Zygote, ZygoteRuleConfig

##

Random.seed!(63);

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

function mysqrt(x::AbstractVector)
    return identity_break_autodiff(sqrt.(abs.(x)))
end

## Various signatures

function make_implicit_sqrt(; kwargs...)
    forward(x) = mysqrt(x)
    conditions(x, y) = abs2.(y) .- abs.(x)
    implicit = ImplicitFunction(forward, conditions; kwargs...)
    return implicit
end

function make_implicit_sqrt_byproduct(; kwargs...)
    forward(x) = one(eltype(x)) .* mysqrt(x), one(eltype(x))
    conditions(x, y, z) = abs2.(y ./ z) .- abs.(x)
    implicit = ImplicitFunction(forward, conditions; kwargs...)
    return implicit
end

function make_implicit_sqrt_args(; kwargs...)
    forward(x, p) = p .* mysqrt(x)
    conditions(x, y, p) = abs2.(y ./ p) .- abs.(x)
    implicit = ImplicitFunction(forward, conditions; kwargs...)
    return implicit
end

function make_implicit_sqrt_kwargs(; kwargs...)
    forward(x; p) = p .* mysqrt(x)
    conditions(x, y; p) = abs2.(y ./ p) .- abs.(x)
    implicit = ImplicitFunction(forward, conditions; kwargs...)
    return implicit
end

## Low level tests

function test_coherent_array_type(a, b)
    @test eltype(a) == eltype(b)
    if a isa Array
        @test b isa Array || b isa (Base.ReshapedArray{T,N,<:Array} where {T,N})
    elseif a isa StaticArray
        @test b isa StaticArray || b isa (Base.ReshapedArray{T,N,<:StaticArray} where {T,N})
    elseif a isa AbstractSparseArray
        @test b isa AbstractSparseArray ||
            b isa (Base.ReshapedArray{T,N,<:AbstractSparseArray} where {T,N})
    else
        error("New array type")
    end
end

function test_implicit_call(x::AbstractVector{T}; type_stability=false, kwargs...) where {T}
    imf1 = make_implicit_sqrt(; kwargs...)
    imf2 = make_implicit_sqrt_byproduct(; kwargs...)
    imf3 = make_implicit_sqrt_args(; kwargs...)
    imf4 = make_implicit_sqrt_kwargs(; kwargs...)

    y_true = mysqrt(x)
    y1 = imf1(x)
    y2, z2 = imf2(x)
    y3 = imf3(x, 1)
    y4 = imf4(x; p=1)

    @testset "Primal value" begin
        @test y1 ≈ y_true
        @test y2 ≈ y_true
        @test y3 ≈ y_true
        @test y4 ≈ y_true
        @test z2 ≈ 1
    end

    @testset "Array type" begin
        test_coherent_array_type(x, y1)
        test_coherent_array_type(x, y2)
        test_coherent_array_type(x, y3)
        test_coherent_array_type(x, y4)
    end

    if type_stability
        @testset "Type stability" begin
            @test_opt target_modules = (ID,) imf1(x)
            @test_opt target_modules = (ID,) imf2(x)
            @test_opt target_modules = (ID,) imf3(x, 1)
            @test_opt target_modules = (ID,) imf4(x; p=1)
        end
    end
end

tag(::AbstractVector{<:ForwardDiff.Dual{T}}) where {T} = T

function test_implicit_duals(
    x::AbstractVector{T}; type_stability=false, kwargs...
) where {T}
    imf1 = make_implicit_sqrt(; kwargs...)
    imf2 = make_implicit_sqrt_byproduct(; kwargs...)
    imf3 = make_implicit_sqrt_args(; kwargs...)
    imf4 = make_implicit_sqrt_kwargs(; kwargs...)

    y_true = mysqrt(x)
    dx = similar(x)
    dx .= 2 * one(T)
    x_and_dx = ForwardDiff.Dual.(x, dx)

    y_and_dy1 = imf1(x_and_dx)
    y_and_dy2, z2 = imf2(x_and_dx)
    y_and_dy3 = imf3(x_and_dx, 1)
    y_and_dy4 = imf4(x_and_dx; p=1)

    @testset "Dual numbers" begin
        @test ForwardDiff.value.(y_and_dy1) ≈ y_true
        @test ForwardDiff.value.(y_and_dy2) ≈ y_true
        @test ForwardDiff.value.(y_and_dy3) ≈ y_true
        @test ForwardDiff.value.(y_and_dy4) ≈ y_true
        @test ForwardDiff.extract_derivative(tag(y_and_dy1), y_and_dy1) ≈
            2 .* inv.(2 .* sqrt.(x))
        @test ForwardDiff.extract_derivative(tag(y_and_dy2), y_and_dy2) ≈
            2 .* inv.(2 .* sqrt.(x))
        @test ForwardDiff.extract_derivative(tag(y_and_dy3), y_and_dy3) ≈
            2 .* inv.(2 .* sqrt.(x))
        @test ForwardDiff.extract_derivative(tag(y_and_dy4), y_and_dy4) ≈
            2 .* inv.(2 .* sqrt.(x))
        @test z2 ≈ 1
    end

    @testset "Array types" begin
        test_coherent_array_type(x, ForwardDiff.value.(y_and_dy1))
        test_coherent_array_type(x, ForwardDiff.value.(y_and_dy2))
        test_coherent_array_type(x, ForwardDiff.value.(y_and_dy3))
        test_coherent_array_type(x, ForwardDiff.value.(y_and_dy4))
    end

    if type_stability
        @testset "Type stability" begin
            @test_opt target_modules = (ID,) imf1(x_and_dx)
            @test_opt target_modules = (ID,) imf2(x_and_dx)
            @test_opt target_modules = (ID,) imf3(x_and_dx, 1)
            @test_opt target_modules = (ID,) imf4(x_and_dx; p=1)
        end
    end
end

function test_implicit_rrule(
    rc, x::AbstractVector{T}; type_stability=false, kwargs...
) where {T}
    imf1 = make_implicit_sqrt(; kwargs...)
    imf2 = make_implicit_sqrt_byproduct(; kwargs...)
    imf3 = make_implicit_sqrt_args(; kwargs...)
    imf4 = make_implicit_sqrt_kwargs(; kwargs...)

    y_true = mysqrt(x)
    dy = similar(y_true)
    dy .= one(eltype(y_true))
    dz = nothing

    y1, pb1 = rrule(rc, imf1, x)
    (y2, z2), pb2 = rrule(rc, imf2, x)
    y3, pb3 = rrule(rc, imf3, x, 1)
    y4, pb4 = rrule(rc, imf4, x; p=1)

    dimf1, dx1 = pb1(dy)
    dimf2, dx2 = pb2((dy, dz))
    dimf3, dx3, dp3 = pb3(dy)
    dimf4, dx4 = pb4(dy)

    @testset "Pullbacks" begin
        @test y1 ≈ y_true
        @test y2 ≈ y_true
        @test y3 ≈ y_true
        @test y4 ≈ y_true
        @test z2 ≈ 1

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
        test_coherent_array_type(x, y1)
        test_coherent_array_type(x, y2)
        test_coherent_array_type(x, y3)
        test_coherent_array_type(x, y4)

        test_coherent_array_type(x, dx1)
        test_coherent_array_type(x, dx2)
        test_coherent_array_type(x, dx3)
        test_coherent_array_type(x, dx4)
    end

    @testset "ChainRulesTestUtils" begin
        test_rrule(rc, imf1, x; atol=1e-2, check_inferred=false)
        test_rrule(rc, imf2, x; atol=5e-2, output_tangent=(dy, 0), check_inferred=false) # see issue https://github.com/gdalle/ImplicitDifferentiation.jl/issues/112
        test_rrule(rc, imf3, x, 1; atol=1e-2, check_inferred=false)
        test_rrule(rc, imf4, x; atol=1e-2, fkwargs=(p=1,), check_inferred=false)
    end

    if type_stability
        @testset "Type stability" begin
            @test_opt target_modules = (ID,) rrule(rc, imf1, x)
            @test_opt target_modules = (ID,) rrule(rc, imf2, x)
            @test_opt target_modules = (ID,) rrule(rc, imf3, x, 1)
            @test_opt target_modules = (ID,) rrule(rc, imf4, x; p=1)

            @test_opt target_modules = (ID,) pb1(dy)
            @test_opt target_modules = (ID,) pb2((dy, dz))
            @test_opt target_modules = (ID,) pb3(dy)
            @test_opt target_modules = (ID,) pb4(dy)
        end
    end
end

## High-level tests per backend

function test_implicit_backend(
    backend::ADTypes.AbstractADType, x::AbstractVector{T}; type_stability=false, kwargs...
) where {T}
    imf1 = make_implicit_sqrt(; kwargs...)
    imf2 = make_implicit_sqrt_byproduct(; kwargs...)
    imf3 = make_implicit_sqrt_args(; kwargs...)
    imf4 = make_implicit_sqrt_kwargs(; kwargs...)

    J1 = DifferentiationInterface.jacobian(imf1, backend, x)
    J2 = DifferentiationInterface.jacobian(first ∘ imf2, backend, x)
    J3 = DifferentiationInterface.jacobian(_x -> imf3(_x, one(eltype(x))), backend, x)

    J4 = if !(backend isa AutoEnzyme)
        DifferentiationInterface.jacobian(_x -> imf4(_x; p=one(eltype(x))), backend, x)
    else
        nothing
    end

    J_true = ForwardDiff.jacobian(_x -> sqrt.(_x), x)

    @testset "Exact Jacobian" begin
        @test J1 ≈ J_true
        @test J2 ≈ J_true
        @test J3 ≈ J_true

        @test eltype(J1) == eltype(x)
        @test eltype(J2) == eltype(x)
        @test eltype(J3) == eltype(x)

        if !(backend isa AutoEnzyme)
            @test J4 ≈ J_true
            @test eltype(J4) == eltype(x)
        end
    end
    return nothing
end

function test_implicit(backends, x; type_stability=false, kwargs...)
    @testset verbose = true "Call" begin
        test_implicit_call(x; kwargs...)
    end
    @testset verbose = true "Duals" begin
        test_implicit_duals(x; kwargs...)
    end
    @testset verbose = true "ChainRule" begin
        test_implicit_rrule(ZygoteRuleConfig(), x; kwargs...)
    end
    @testset "$backend" for backend in backends
        test_implicit_backend(backend, x; kwargs...)
    end
    return nothing
end

## Parameter combinations

backends = [
    AutoForwardDiff(; chunksize=1), #
    AutoEnzyme(Enzyme.Forward),
    AutoZygote(),
]

linear_solver_candidates = (
    \, #
    # ID.DefaultLinearSolver(),
)

conditions_backend_candidates = (
    nothing, #
    # AutoForwardDiff(; chunksize=1),
    # AutoEnzyme(Enzyme.Forward),
);

x_candidates = (
    rand(Float32, 2), #
);

## Test loop

@testset verbose = true "$(typeof(x)) - $linear_solver - $(typeof(conditions_backend))" for (
    x, linear_solver, conditions_backend
) in Iterators.product(
    x_candidates, linear_solver_candidates, conditions_backend_candidates
)
    test_implicit(
        backends,
        x;
        linear_solver,
        conditions_x_backend=conditions_backend,
        conditions_y_backend=conditions_backend,
    )
end;
