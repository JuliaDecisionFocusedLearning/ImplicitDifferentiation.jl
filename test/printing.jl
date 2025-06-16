using TestItems

@testitem "Settings" begin
    @test startswith(string(OperatorRepresentation()), "Operator")
    @test startswith(string(IterativeLinearSolver(; atol=1e-5)), "Iterative")
    @test startswith(string(IterativeLinearSolver()), "Iterative")
end
