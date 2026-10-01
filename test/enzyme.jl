using TestItems

@testitem "Enzyme rules" setup = [TestUtils] begin
    using .TestUtils
    # `default_conditions` triggers an Enzyme bug with inner forward mode (see
    # `enzyme_broken` in test/utils.jl), so these conditions avoid it in order to test
    # all rules with all linear solvers.
    for (linear_solver, representation) in [
        (DirectLinearSolver(), MatrixRepresentation()),
        (IterativeLinearSolver(), MatrixRepresentation()),
        (IterativeLinearSolver(), OperatorRepresentation()),
        (IterativeLeastSquaresSolver(), OperatorRepresentation()),
    ]
        yield()
        scen = Scenario(;
            solver=x -> (sqrt.(x), nothing),
            conditions=(x, y, z) -> y .^ 2 .- x,
            x=float.(1:3),
            implicit_kwargs=(; representation, linear_solver),
        )
        test_implicit_enzyme(scen)
        test_implicit_enzyme(add_arg_mult(scen))
    end
end
