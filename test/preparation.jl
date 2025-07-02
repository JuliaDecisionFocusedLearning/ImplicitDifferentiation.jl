@testitem "Preparation" begin
    using ImplicitDifferentiation
    using ImplicitDifferentiation:
        prepare_implicit, build_A, build_Aᵀ, build_B, build_Bᵀ, JVP, VJP
    using ADTypes
    using ADTypes: ForwardOrReverseMode, ForwardMode, ReverseMode
    using ForwardDiff: ForwardDiff
    using LinearAlgebra: Factorization, TransposeFactorization
    using Zygote: Zygote
    using Test

    const GenericMatrix = Union{AbstractMatrix,Factorization,TransposeFactorization}

    solver(x) = sqrt.(x), nothing
    conditions(x, y, z) = y .^ 2 .- x

    implicit = ImplicitFunction(
        solver,
        conditions;
        backends=(; x=AutoForwardDiff(), y=AutoForwardDiff()),
        representation=MatrixRepresentation(),
    )
    implicit_iterative = ImplicitFunction(
        solver, conditions; backends=(; x=AutoForwardDiff(), y=AutoForwardDiff())
    )
    implicit_nobackends = ImplicitFunction(solver, conditions)

    x = rand(5)
    y, z = implicit(x)
    c = conditions(x, y, z)
    suggested_backend = AutoEnzyme()

    @testset "None" begin
        prep = prepare_implicit(ForwardOrReverseMode(), implicit_nobackends, x)
        @test prep.prep_A === nothing
        @test prep.prep_Aᵀ === nothing
        @test prep.prep_B === nothing
        @test prep.prep_Bᵀ === nothing
    end

    @testset "ForwardMode" begin
        prep = prepare_implicit(ForwardMode(), implicit, x)
        @test prep.prep_A !== nothing
        @test prep.prep_Aᵀ === nothing
        @test prep.prep_B !== nothing
        @test prep.prep_Bᵀ === nothing
        @test build_A(implicit, prep, x, y, z, c; suggested_backend) isa GenericMatrix
        @test build_Aᵀ(implicit, prep, x, y, z, c; suggested_backend) isa GenericMatrix
        @test build_B(implicit, prep, x, y, z, c; suggested_backend) isa JVP
        @test build_Bᵀ(implicit, prep, x, y, z, c; suggested_backend) isa VJP
    end

    @testset "ReverseMode" begin
        prep = prepare_implicit(ReverseMode(), implicit, x)
        @test prep.prep_A === nothing
        @test prep.prep_Aᵀ !== nothing
        @test prep.prep_B === nothing
        @test prep.prep_Bᵀ !== nothing
        @test build_A(implicit, prep, x, y, z, c; suggested_backend) isa GenericMatrix
        @test build_Aᵀ(implicit, prep, x, y, z, c; suggested_backend) isa GenericMatrix
        @test build_B(implicit, prep, x, y, z, c; suggested_backend) isa JVP
        @test build_Bᵀ(implicit, prep, x, y, z, c; suggested_backend) isa VJP
    end

    @testset "Both" begin
        prep = prepare_implicit(ForwardOrReverseMode(), implicit_iterative, x)
        @test prep.prep_A !== nothing
        @test prep.prep_Aᵀ !== nothing
        @test prep.prep_B !== nothing
        @test prep.prep_Bᵀ !== nothing
        @test build_A(implicit_iterative, prep, x, y, z, c; suggested_backend) isa JVP
        @test build_Aᵀ(implicit_iterative, prep, x, y, z, c; suggested_backend) isa VJP
        @test build_B(implicit_iterative, prep, x, y, z, c; suggested_backend) isa JVP
        @test build_Bᵀ(implicit_iterative, prep, x, y, z, c; suggested_backend) isa VJP
    end
end
