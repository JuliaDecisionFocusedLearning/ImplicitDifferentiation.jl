using TestItems

@testitem "Matrix" setup = [TestUtils] begin
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
                linear_solver=\,
                backends,
                preparation,
                input_example=(x,),
            ),
        )
        scen2 = add_arg_mult(scen)
        @info "$scen"
        test_implicit(scen)
        test_implicit(scen2)
    end
end;

@testitem "Krylov" setup = [TestUtils] begin
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
                linear_solver=IterativeLinearSolver{:Krylov}(),
                backends,
                preparation,
                input_example=(x,),
            ),
        )
        @info "$scen"
        scen2 = add_arg_mult(scen)
        test_implicit(scen)
        test_implicit(scen2)
    end
end;

@testitem "IterativeSolvers" setup = [TestUtils] begin
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
                representation=OperatorRepresentation{:LinearMap}(),
                linear_solver=IterativeLinearSolver{:IterativeSolvers}(),
                backends,
                preparation,
                input_example=(x,),
            ),
        )
        @info "$scen"
        test_implicit(scen)
    end
end;
