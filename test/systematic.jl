using TestItems

@testitem "Direct" setup = [TestUtils] begin
    using ADTypes, .TestUtils
    for (backends, preparation, x) in Iterators.product(
        [nothing, (; x=AutoForwardDiff(), y=AutoZygote())],
        [nothing, ADTypes.ForwardOrReverseMode()],
        [float.(1:3), reshape(float.(1:6), 3, 2)],
    )
        yield()
        scen = Scenario(;
            solver=default_solver,
            conditions=default_conditions,
            x=x,
            implicit_kwargs=(;
                representation=MatrixRepresentation(),
                linear_solver=DirectLinearSolver(),
                backends,
                preparation,
                input_example=(x,),
            ),
        )
        scen2 = add_arg_mult(scen)
        test_implicit(scen)
        test_implicit(scen2)
    end
end;

@testitem "Iterative" setup = [TestUtils] begin
    using ADTypes, .TestUtils
    for (backends, preparation, x) in Iterators.product(
        [nothing, (; x=AutoForwardDiff(), y=AutoZygote())],
        [nothing, ADTypes.ForwardOrReverseMode()],
        [float.(1:3), reshape(float.(1:6), 3, 2)],
    )
        yield()
        scen = Scenario(;
            solver=default_solver,
            conditions=default_conditions,
            x=x,
            implicit_kwargs=(;
                representation=OperatorRepresentation{:LinearOperators}(),
                linear_solver=IterativeLinearSolver(),
                backends,
                preparation,
                input_example=(x,),
            ),
        )
        scen2 = add_arg_mult(scen)
        test_implicit(scen)
        test_implicit(scen2)
    end
end;

@testitem "ComponentVector" setup = [TestUtils] begin
    using .TestUtils
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
