using TestItems

@testitem "Settings" begin
    @test contains(string(ImplicitFunction(nothing, nothing)), "ImplicitFunction")
    @test contains(string(IterativeLinearSolver()), "IterativeLinearSolver")
    @test contains(string(IterativeLinearSolver(; rtol=1e-3)), "IterativeLinearSolver")
end
