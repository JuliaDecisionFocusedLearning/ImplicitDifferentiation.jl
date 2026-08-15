using TestItems

@testitem "Enzyme rules" begin
    using Enzyme
    using EnzymeTestUtils
    using ImplicitDifferentiation
    using ImplicitDifferentiation:
        ImplicitFunction,
        DirectLinearSolver,
        IterativeLeastSquaresSolver,
        MatrixRepresentation,
        OperatorRepresentation

    # `vcat` triggers an upstream Enzyme/DifferentiationInterface bug (see the comment in
    # `test_implicit_jacobian` in test/utils.jl), so these scenarios avoid it entirely.
    # That lets EnzymeTestUtils exercise the forward rule as well as the reverse one.
    solver(x) = sqrt.(x), nothing
    conditions(x, y, z) = y .^ 2 .- x

    solver_arg(x, a) = sqrt.(x) .* a, nothing
    conditions_arg(x, y, z, a) = (y ./ a) .^ 2 .- x

    x = float.(1:3)

    @testset "$linear_solver" for (linear_solver, representation) in [
        (DirectLinearSolver(), MatrixRepresentation()),
        (IterativeLeastSquaresSolver(), OperatorRepresentation()),
    ]
        implicit = ImplicitFunction(solver, conditions; representation, linear_solver)
        implicit_arg = ImplicitFunction(
            solver_arg, conditions_arg; representation, linear_solver
        )

        @testset "Forward" begin
            for Tret in (Duplicated, DuplicatedNoNeed)
                test_forward(implicit, Tret, (x, Duplicated))
            end
            for Tret in (BatchDuplicated, BatchDuplicatedNoNeed)
                test_forward(implicit, Tret, (x, BatchDuplicated))
            end
            test_forward(implicit_arg, Duplicated, (x, Duplicated), 2.0)
        end

        @testset "Reverse" begin
            test_reverse(implicit, Duplicated, (x, Duplicated))
            test_reverse(implicit, BatchDuplicated, (x, BatchDuplicated))
            test_reverse(implicit_arg, Duplicated, (x, Duplicated), 2.0)
        end
    end
end
