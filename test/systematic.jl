using TestItems

@testitem "Direct" setup = [TestUtils] begin
    using ADTypes, .TestUtils
    for (backends, x) in
        Iterators.product([nothing, (; x=AutoForwardDiff(), y=AutoZygote())], [float.(1:3)])
        yield()
        scen = Scenario(;
            solver=default_solver,
            conditions=default_conditions,
            x=x,
            implicit_kwargs=(;
                representation=MatrixRepresentation(),
                linear_solver=DirectLinearSolver(),
                backends,
            ),
        )
        scen2 = add_arg_mult(scen)
        test_implicit(scen)
        test_implicit(scen2)
    end
end;

@testitem "Iterative" setup = [TestUtils] begin
    using ADTypes, .TestUtils
    for (backends, x) in Iterators.product(
        [nothing, (; x=AutoForwardDiff(), y=AutoZygote())],
        [float.(1:3), reshape(float.(1:6), 3, 2)],
    )
        yield()
        scen = Scenario(;
            solver=default_solver,
            conditions=default_conditions,
            x=x,
            implicit_kwargs=(;
                representation=OperatorRepresentation(),
                linear_solver=IterativeLinearSolver(),
                backends,
            ),
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
        implicit_kwargs=(; strict=Val(false)),
    )
    scen2 = add_arg_mult(scen)
    test_implicit(scen)
    test_implicit(scen2)
end;
