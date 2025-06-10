using TestItems

@testitem "Systematic tests" begin
    using ADTypes
    using ADTypes: ForwardMode, ReverseMode
    using ForwardDiff: ForwardDiff
    using ImplicitDifferentiation
    using Test
    using Zygote: Zygote, ZygoteRuleConfig
    using FiniteDiff: FiniteDiff

    include("utils.jl")

    ## Parameter combinations

    representation_linear_solver_candidates = [
        (MatrixRepresentation(), \),  #
        (OperatorRepresentation{:LinearOperators}(), IterativeLinearSolver{:Krylov}()),
        (OperatorRepresentation{:LinearMaps}(), IterativeLinearSolver{:IterativeSolvers}()),
    ]
    backend_candidates = [nothing, AutoForwardDiff(), AutoZygote()]
    preparation_candidates = [nothing, ForwardMode(), ReverseMode()]
    x_candidates = [float.(1:6), reshape(float.(1:12), 6, 2)]

    ## Test loop

    @testset for (representation, linear_solver) in representation_linear_solver_candidates
        for (backend, preparation, x) in
            Iterators.product(backend_candidates, preparation_candidates, x_candidates)
            x_type = typeof(x)
            @info "Testing" linear_solver backend representation preparation x_type
            if (representation isa OperatorRepresentation && linear_solver == \)
                continue
            end
            outer_backends = [AutoForwardDiff(), AutoZygote()]
            x = Float64.(1:6)
            @testset "$((; linear_solver, backend, preparation, x_type))" begin
                test_implicit(
                    outer_backends,
                    x;
                    representation,
                    backends=isnothing(backend) ? nothing : (; x=backend, y=backend),
                    preparation,
                    linear_solver,
                    strict=if linear_solver isa IterativeLinearSolver{:IterativeSolvers}
                        Val(false)
                    else
                        Val(true)
                    end,
                )
            end
        end
    end
end;
