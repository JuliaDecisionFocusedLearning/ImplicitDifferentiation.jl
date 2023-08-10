module ImplicitDifferentiationZygoteExt

@static if isdefined(Base, :get_extension)
    using Zygote: jacobian
else
    using ..Zygote: jacobian
end

using ImplicitDifferentiation: ImplicitFunction, identity_break_autodiff
using ImplicitDifferentiation: DirectLinearSolver, IterativeLinearSolver
using PrecompileTools: @compile_workload

@compile_workload begin
    forward(x) = sqrt.(identity_break_autodiff(x))
    conditions(x, y) = y .^ 2 .- x
    for linear_solver in (DirectLinearSolver(), IterativeLinearSolver())
        implicit = ImplicitFunction(forward, conditions; linear_solver)
        x = rand(2)
        implicit(x)
        jacobian(implicit, x)
    end
end

end
