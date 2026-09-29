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

@testitem "Mooncake other inputs" setup = [TestUtils] begin
    using ADTypes, LinearAlgebra, .TestUtils
    using .TestUtils: NonDifferentiable
    import DifferentiationInterface as DI
    x = float.(1:3)
    a = float.(4:6)
    explicit(x, a) = sqrt.(x .* a)
    conditions(x, y, z, a) = y .^ 2 .- x .* a
    implicit = ImplicitFunction(
        NonDifferentiable((x, a) -> (explicit(x, a), nothing)), conditions
    )
    function captured(a)
        solver = NonDifferentiable(x -> (explicit(x, a), nothing))
        return ImplicitFunction(solver, (x, y, z) -> y .^ 2 .- x .* a)
    end
    implicit_scalar_z = ImplicitFunction(
        NonDifferentiable((x, a) -> (explicit(x, a), 0.0)), conditions
    )
    implicit_forwarddiff = ImplicitFunction(
        NonDifferentiable((x, a) -> (explicit(x, a), nothing)),
        conditions;
        backends=(; x=AutoForwardDiff(), y=AutoForwardDiff()),
    )
    jac_a = DI.jacobian(a -> explicit(x, a), AutoForwardDiff(), a)
    jac_x = DI.jacobian(x -> explicit(x, 2.0), AutoForwardDiff(), x)
    @testset "$backend" for backend in [
        AutoMooncakeForward(; config=nothing), AutoMooncake(; config=nothing)
    ]
        @test DI.jacobian(a -> first(implicit(x, a)), backend, a) ≈ jac_a
        @test DI.jacobian(a -> first(captured(a)(x)), backend, a) ≈ jac_a
        @test DI.jacobian(x -> first(implicit(x, 2.0)), backend, x) ≈ jac_x
        # a `z` with nonzero rdata
        @test DI.jacobian(x -> first(implicit_scalar_z(x, 2.0)), backend, x) ≈ jac_x
        # an `x` whose Mooncake tangent is not an array
        @test DI.jacobian(x -> first(implicit_forwarddiff(view(x, :), 2.0)), backend, x) ≈
            jac_x
        if backend isa AutoMooncake
            # DI's Mooncake forward mode rejects an array tangent for a `SubArray`
            @test DI.jacobian(x -> first(implicit(view(x, :), 2.0)), backend, x) ≈ jac_x
        end
    end
end;
