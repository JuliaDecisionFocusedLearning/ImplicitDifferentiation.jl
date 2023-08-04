using ForwardDiff
using ImplicitDifferentiation
using Test
using Zygote

@testset "Byproduct handling" begin
    f = (_) -> [1.0, 2.0]
    c = (_, _) -> [0.0, 0.0]
    imf1 = ImplicitFunction(f, c)
    @test_throws ArgumentError imf1(zeros(2))
    f = (_) -> [1.0, 2.0, 3.0]
    imf2 = ImplicitFunction(f, c)
    @test_throws ArgumentError imf2(zeros(2))
end

@testset "Only accept one array" begin
    f = (_) -> [1.0]
    c = (_, _) -> [0.0]
    imf = ImplicitFunction(f, c)
    @test_throws MethodError imf("hello")
    @test_throws MethodError imf([1.0], [1.0])
end

@testset verbose = true "Derivative NaNs" begin
    x = zeros(Float32, 2)
    linear_solvers = (IterativeLinearSolver(), DirectLinearSolver())
    @testset "Infinite derivative" begin
        f = x -> sqrt.(x)  # nondifferentiable at 0
        c = (x, y) -> y .^ 2 .- x
        for linear_solver in linear_solvers
            @testset "$(typeof(linear_solver))" begin
                implicit = ImplicitFunction(f, c; linear_solver)
                J1 = ForwardDiff.jacobian(implicit, x)
                J2 = Zygote.jacobian(implicit, x)[1]
                @test all(isnan, J1) && eltype(J1) == Float32
                @test all(isnan, J2) && eltype(J2) == Float32
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
                @test all(isnan, J1) && eltype(J1) == Float32
                @test all(isnan, J2) && eltype(J2) == Float32
            end
        end
    end
end
