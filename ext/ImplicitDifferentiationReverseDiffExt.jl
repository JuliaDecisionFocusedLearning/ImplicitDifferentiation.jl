module ImplicitDifferentiationReverseDiffExt

@static if isdefined(Base, :get_extension)
    using ReverseDiff: TrackedArray, TrackedReal, @grad_from_chainrules, jacobian
else
    using ..ReverseDiff: TrackedArray, TrackedReal, @grad_from_chainrules, jacobian
end

using AbstractDifferentiation: ReverseDiffBackend
using ChainRulesCore: ChainRulesCore, RuleConfig, HasReverseMode, NoForwardsMode, rrule
using ImplicitDifferentiation:
    ImplicitDifferentiation,
    ImplicitFunction,
    DirectLinearSolver,
    IterativeLinearSolver,
    call_implicit_function,
    check_valid_output,
    identity_break_autodiff
using LinearAlgebra: mul!
using PrecompileTools: @compile_workload

struct MyReverseDiffRuleConfig <: RuleConfig{Union{HasReverseMode,NoForwardsMode}} end

function ImplicitDifferentiation.reverse_conditions_backend(
    ::MyReverseDiffRuleConfig, ::ImplicitFunction{F,C,L,Nothing}
) where {F,C,L}
    return ReverseDiffBackend()
end

function ChainRulesCore.rrule(
    ::typeof(call_implicit_function),
    implicit::ImplicitFunction,
    x::AbstractArray,
    args...;
    kwargs...,
)
    # The macro ReverseDiff.@grad_from_chainrules calls ChainRulesCore.rrule without a ruleconfig
    rc = MyReverseDiffRuleConfig()
    return rrule(rc, call_implicit_function, implicit, x, args...; kwargs...)
end

@grad_from_chainrules ImplicitDifferentiation.call_implicit_function(
    implicit::ImplicitFunction, x::TrackedArray
)

@compile_workload begin
    forward(x) = sqrt.(identity_break_autodiff(x))
    conditions(x, y) = y .^ 2 .- x
    for linear_solver in (DirectLinearSolver(), IterativeLinearSolver())
        implicit = ImplicitFunction(forward, conditions; linear_solver)
        x = rand(2)
        implicit(x)
        # TODO: the following line kills Julia during precompilation, so fast that I cannot see the error
        # jacobian(_x -> call_implicit_function(implicit, _x), x)  
    end
end

end
