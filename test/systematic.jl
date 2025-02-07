using ADTypes
using Enzyme: Enzyme
using ForwardDiff: ForwardDiff
using SparseArrays
using StaticArrays
using Test
using Zygote: Zygote, ZygoteRuleConfig

include("utils.jl")

## Parameter combinations

backends = [
    AutoForwardDiff(; chunksize=1), #
    AutoEnzyme(;
        mode=Enzyme.set_runtime_activity(Enzyme.Forward), function_annotation=Enzyme.Const
    ),
    AutoZygote(),
]

linear_solver_candidates = (
    \, #
    ID.KrylovLinearSolver(),
)

conditions_backend_candidates = (
    nothing, #
    AutoForwardDiff(; chunksize=1),
    AutoEnzyme(;
        mode=Enzyme.set_runtime_activity(Enzyme.Forward), function_annotation=Enzyme.Const
    ),
);

x_candidates = (
    Float32[3, 4], #
    # MVector{2}(Float32[3, 4]), #
);

## Test loop

@testset verbose = false "$(typeof(x)) - $linear_solver - $(typeof(conditions_backend))" for (
    x, linear_solver, conditions_backend
) in Iterators.product(
    x_candidates, linear_solver_candidates, conditions_backend_candidates
)
    if x isa StaticArray && (linear_solver != \)
        continue
    end
    @info "Testing $(typeof(x)) - $linear_solver - $(typeof(conditions_backend))"
    test_implicit(
        backends,
        x;
        linear_solver,
        conditions_x_backend=conditions_backend,
        conditions_y_backend=conditions_backend,
    )
end;
