using ADTypes
using ForwardDiff: ForwardDiff
using Test
using Zygote: Zygote, ZygoteRuleConfig
using FiniteDiff: FiniteDiff

include("utils.jl")

## Parameter combinations

backend_candidates = [AutoForwardDiff(), AutoZygote()]
linear_solver_candidates = [ID.KrylovLinearSolver()]
lazy_candidates = [true, false]

## Test loop

@testset for (backend, linear_solver, lazy) in Iterators.product(
    backend_candidates, linear_solver_candidates, lazy_candidates
)
    @info "Testing $backend - $linear_solver - $lazy"
    outer_backends = [AutoForwardDiff(), AutoZygote()]
    x = Float64.(1:6)
    test_implicit(outer_backends, x; backend, linear_solver, lazy)
end;
