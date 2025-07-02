using TestItems

@testitem "Settings" begin
    @test startswith(string(ImplicitFunction(nothing, nothing)), "ImplicitFunction")
    @test startswith(string(IterativeLinearSolver()), "IterativeLinearSolver")
    @test startswith(string(IterativeLinearSolver(; rtol=1e-3)), "IterativeLinearSolver")
end
