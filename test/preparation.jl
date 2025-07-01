@testitem "Preparation" begin
    using ImplicitDifferentiation
    using ADTypes
    using ADTypes: ForwardOrReverseMode, ForwardMode, ReverseMode
    using ForwardDiff: ForwardDiff
    using Zygote: Zygote
    using Test

    solver(x) = sqrt.(x), nothing
    conditions(x, y, z) = y .^ 2 .- x
    implicit = ImplicitFunction(
        solver, conditions; backends=(; x=AutoForwardDiff(), y=AutoForwardDiff())
    )
    implicit_nobackends = ImplicitFunction(solver, conditions)
    x = rand(5)

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
    end

    @testset "ReverseMode" begin
        prep = prepare_implicit(ReverseMode(), implicit, x)
        @test prep.prep_A === nothing
        @test prep.prep_Aᵀ !== nothing
        @test prep.prep_B === nothing
        @test prep.prep_Bᵀ !== nothing
    end

    @testset "Both" begin
        prep = prepare_implicit(ForwardOrReverseMode(), implicit, x)
        @test prep.prep_A !== nothing
        @test prep.prep_Aᵀ !== nothing
        @test prep.prep_B !== nothing
        @test prep.prep_Bᵀ !== nothing
    end
end
