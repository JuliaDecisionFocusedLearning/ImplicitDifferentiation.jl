using TestItems

@testitem "Matrix" setup = [TestUtils] begin
    using ADTypes, .TestUtils
    representation = MatrixRepresentation()
    for (linear_solver, backends, x) in Iterators.product(
        [DirectLinearSolver(), IterativeLinearSolver()],
        [nothing, (; x=AutoForwardDiff(), y=AutoZygote())],
        [float.(1:3)],
    )
        yield()
        scen = Scenario(;
            solver=default_solver,
            conditions=default_conditions,
            x=x,
            implicit_kwargs=(; representation, linear_solver, backends),
        )
        scen2 = add_arg_mult(scen)
        test_implicit(scen)
        test_implicit(scen2)
    end
end;

@testitem "Operator" setup = [TestUtils] begin
    using ADTypes, .TestUtils
    representation = OperatorRepresentation()
    for (linear_solver, backends, x) in Iterators.product(
        [
            IterativeLinearSolver(),
            IterativeLinearSolver(; rtol=1e-8),
            IterativeLinearSolver(; issymmetric=true, isposdef=true),
            IterativeLeastSquaresSolver(),
        ],
        [nothing, (; x=AutoForwardDiff(), y=AutoZygote())],
        [float.(1:3), reshape(float.(1:6), 3, 2)],
    )
        yield()
        scen = Scenario(;
            solver=default_solver,
            conditions=default_conditions,
            x=x,
            implicit_kwargs=(; representation, linear_solver, backends),
        )
        scen2 = add_arg_mult(scen)
        test_implicit(scen; type_stability=VERSION >= v"1.11")
        test_implicit(scen2; type_stability=VERSION >= v"1.11")
    end
end;

@testitem "ComponentVector" setup = [TestUtils] begin
    using ComponentArrays, .TestUtils
    x = ComponentVector(; a=float.(1:3), b=float.(4:6))
    scen = Scenario(;
        solver=default_solver,
        conditions=default_conditions,
        x=x,
        implicit_kwargs=(; linear_solver=IterativeLeastSquaresSolver()),
    )
    scen2 = add_arg_mult(scen)
    test_implicit(scen)
    test_implicit(scen2)
end;
