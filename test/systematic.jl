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

    # Test for output vector of length 1
    for (linear_solver, backends) in Iterators.product(
        [DirectLinearSolver(), IterativeLinearSolver()],
        [nothing, (; x=AutoForwardDiff(), y=AutoZygote())],
    )
        yield()
        scen = Scenario(;
            solver=x -> (sqrt.(x), nothing),
            conditions=(x, y, z) -> y .^ 2 .- x,
            x=[1.0],
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
    using ADTypes, ComponentArrays, .TestUtils
    x = ComponentVector(; a=float.(1:3), b=float.(4:6))
    scen = Scenario(;
        solver=default_solver,
        conditions=default_conditions,
        x=x,
        implicit_kwargs=(; linear_solver=IterativeLeastSquaresSolver()),
    )
    scen2 = add_arg_mult(scen)
    # the Mooncake rules take `Array` inputs only
    test_implicit(scen, [AutoForwardDiff(), AutoZygote()])
    test_implicit(scen2, [AutoForwardDiff(), AutoZygote()])
end;

@testitem "Mooncake scalar byproduct" setup = [TestUtils] begin
    using ADTypes, Mooncake, .TestUtils
    using Random: Xoshiro
    using .TestUtils: NonDifferentiable
    import DifferentiationInterface as DI
    x = float.(1:3)
    explicit(x, a) = sqrt.(x .* a)
    conditions(x, y, z, a) = y .^ 2 .- x .* a
    # a `z` with nonzero rdata
    implicit = ImplicitFunction(
        NonDifferentiable((x, a) -> (explicit(x, a), 0.0)), conditions
    )
    jac_x = DI.jacobian(x -> explicit(x, 2), AutoForwardDiff(), x)
    @testset "$backend" for backend in [
        AutoMooncakeForward(; config=nothing), AutoMooncake(; config=nothing)
    ]
        @test DI.jacobian(x -> first(implicit(x, 2)), backend, x) ≈ jac_x
    end
    rng = Xoshiro(0)
    modes = (Mooncake.ForwardMode, Mooncake.ReverseMode)
    @testset "Mooncake rule - $mode" for mode in modes
        Mooncake.TestUtils.test_rule(
            rng, implicit, x, 2; is_primitive=true, mode, unsafe_perturb=true
        )
    end
end;
