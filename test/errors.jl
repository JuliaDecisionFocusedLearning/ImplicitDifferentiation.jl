using ChainRulesCore
using ChainRulesTestUtils
using ForwardDiff
using ImplicitDifferentiation
using ImplicitDifferentiation: call_implicit_function
using Test
using Zygote

@testset "Byproduct handling" begin
    f1 = (_) -> (1, 2)
    f2 = (_) -> ([1.0], 2, 3)
    c = nothing
    imf1 = ImplicitFunction(f1, c)
    imf2 = ImplicitFunction(f2, c)
    @test_throws DimensionMismatch imf1(zeros(1))
    @test_throws DimensionMismatch imf2(zeros(1))
end

@testset "Only accept one array" begin
    f = identity
    c = nothing
    imf = ImplicitFunction(f, c)
    @test_throws MethodError imf((1.0,))
    @test_throws MethodError imf([1.0], [1.0])
end

@testset verbose = true "Derivative NaNs" begin
    x = zeros(Float32, 2)
    linear_solvers = (
        IterativeLinearSolver(; verbose=false),  #
        IterativeLinearSolver(; verbose=false, accept_inconsistent=true),  #
        DirectLinearSolver(; verbose=false),  #
    )
    function should_give_nan(linear_solver)
        return linear_solver isa DirectLinearSolver || !linear_solver.accept_inconsistent
    end

    @testset "Infinite derivative" begin
        f = x -> sqrt.(x)  # nondifferentiable at 0
        c = (x, y) -> y .^ 2 .- x
        for linear_solver in linear_solvers
            @testset "$(typeof(linear_solver))" begin
                implicit = ImplicitFunction(f, c; linear_solver)
                J1 = ForwardDiff.jacobian(implicit, x)
                J2 = Zygote.jacobian(implicit, x)[1]
                @test all(isnan, J1) == should_give_nan(linear_solver)
                @test all(isnan, J2) == should_give_nan(linear_solver)
                @test eltype(J1) == Float32
                @test eltype(J2) == Float32
            end
        end
    end

    @testset "Singular linear system" begin
        f = x -> x  # wrong solver
        c = (x, y) -> (x .+ 1) .^ 2 .- y .^ 2
        for linear_solver in linear_solvers
            @testset "$(typeof(linear_solver))" begin
                implicit = ImplicitFunction(f, c; linear_solver)
                J1 = ForwardDiff.jacobian(implicit, x)
                J2 = Zygote.jacobian(implicit, x)[1]
                @test all(isnan, J1) == should_give_nan(linear_solver)
                @test all(isnan, J2) == should_give_nan(linear_solver)
                @test eltype(J1) == Float32
                @test eltype(J2) == Float32
            end
        end
    end
end

@testset "Weird ChainRulesTestUtils behavior" begin
    x = rand(3)
    forward(x) = sqrt.(abs.(x)), 1
    conditions(x, y, z) = abs.(y ./ z) .- abs.(x)
    implicit = ImplicitFunction(forward, conditions)
    y, z = implicit(x)
    dy = similar(y)
    rc = Zygote.ZygoteRuleConfig()
    test_rrule(rc, call_implicit_function, implicit, x; atol=1e-2, output_tangent=(dy, 0))
    @test_skip test_rrule(
        rc,
        call_implicit_function,
        implicit,
        x;
        atol=1e-2,
        output_tangent=(dy, NoTangent()),
    )
end
