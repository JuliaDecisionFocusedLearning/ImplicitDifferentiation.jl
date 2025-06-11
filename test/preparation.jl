@testitem "Preparation" begin
    using ImplicitDifferentiation
    using ADTypes
    using ADTypes: ForwardOrReverseMode, ForwardMode, ReverseMode
    using ForwardDiff: ForwardDiff
    using Zygote: Zygote
    using Test

    solver(x) = sqrt.(x), nothing
    conditions(x, y, z) = y .^ 2 .- x
    x = rand(5)
    input_example = (x,)

    @testset "None" begin
        implicit_nones = ImplicitFunction(solver, conditions)
        @test implicit_none.prep_A === nothing
        @test implicit_none.prep_Aᵀ === nothing
        @test implicit_none.prep_B === nothing
        @test implicit_none.prep_Bᵀ === nothing
    end

    @testset "ForwardMode" begin
        implicit_forward = ImplicitFunction(
            solver,
            conditions;
            preparation=ForwardMode(),
            backends=(; x=AutoForwardDiff(), y=AutoForwardDiff()),
            input_example,
        )
        @test implicit_forward.prep_A !== nothing
        @test implicit_forward.prep_Aᵀ === nothing
        @test implicit_forward.prep_B !== nothing
        @test implicit_forward.prep_Bᵀ === nothing
    end

    @testset "ReverseMode" begin
        implicit_reverse = ImplicitFunction(
            solver,
            conditions;
            preparation=ReverseMode(),
            backends=(; x=AutoZygote(), y=AutoZygote()),
            input_example,
        )
        @test implicit_reverse.prep_A === nothing
        @test implicit_reverse.prep_Aᵀ !== nothing
        @test implicit_reverse.prep_B === nothing
        @test implicit_reverse.prep_Bᵀ !== nothing
    end

    @testset "Both" begin
        implicit_both = ImplicitFunction(
            solver,
            conditions;
            preparation=ForwardOrReverseMode(),
            backends=(; x=AutoForwardDiff(), y=AutoZygote()),
            input_example,
        )
        @test implicit_both.prep_A !== nothing
        @test implicit_both.prep_Aᵀ !== nothing
        @test implicit_both.prep_B !== nothing
        @test implicit_both.prep_Bᵀ !== nothing
    end
end
