using ADTypes
using ADTypes: ForwardMode, ReverseMode
using ForwardDiff: ForwardDiff
using ImplicitDifferentiation
using Test
using Zygote: Zygote, ZygoteRuleConfig
using FiniteDiff: FiniteDiff

include("utils.jl")

## Parameter combinations

linear_solver_candidates = [ID.IterativeLinearSolver(), \]
representation_candidates = [MatrixRepresentation(), OperatorRepresentation()]
backend_candidates = [nothing, AutoForwardDiff(), AutoZygote()]
preparation_candidates = [nothing, ForwardMode(), ReverseMode()]
x_candidates = [float.(1:6), reshape(float.(1:12), 6, 2)]

## Test loop

@testset verbose = true "Systematic tests" begin
    @testset for representation in representation_candidates
        for (linear_solver, backend, preparation, x) in Iterators.product(
            linear_solver_candidates,
            backend_candidates,
            preparation_candidates,
            x_candidates,
        )
            x_type = typeof(x)
            @info "Testing" linear_solver backend representation preparation x_type
            if (representation isa OperatorRepresentation && linear_solver == \)
                continue
            end
            outer_backends = [AutoForwardDiff(), AutoZygote()]
            x = Float64.(1:6)
            @testset "$((; linear_solver, backend, preparation, x_type))" begin
                test_implicit(
                    outer_backends, x; representation, backend, preparation, linear_solver
                )
            end
        end
    end
end;
