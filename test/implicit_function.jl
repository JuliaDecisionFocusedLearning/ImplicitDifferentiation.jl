@testset verbose = true "Explicit formulae" begin
    implicit = ImplicitFunction(;
        forward=identity, conditions=(x, y) -> y - x, linear_solver=cg
    )

    d = 100
    x = rand(d)
    v = rand(d)

    _, pullback_implicit = rrule_via_ad(Zygote.ZygoteRuleConfig(), implicit, x)

    @test mean(abs, implicit(x) - x) < 1e-3
    @test mean(abs, pullback_implicit(v)[2] - v) < 1e-3
end

@testset verbose = true "Black box" begin
    square_distance(y, x) = 0.5 * sum(abs2, y - x)

    function forward(x)
        fun = OptimizationFunction(square_distance, GalacticOptim.AutoForwardDiff())
        prob = OptimizationProblem(fun, zero(x), x)
        sol = solve(prob, LBFGS())
        return sol.u
    end

    function conditions(x, y)
        gs = Zygote.gradient(ỹ -> square_distance(ỹ, x), y)
        return gs[1]
    end

    implicit_black_box = ImplicitFunction(;
        forward=forward, conditions=conditions, linear_solver=cg
    )

    d = 100
    x = rand(d)
    v = rand(d)

    _, pullback_implicit_black_box = rrule_via_ad(
        Zygote.ZygoteRuleConfig(), implicit_black_box, x
    )

    @test mean(abs, implicit_black_box(x) - x) < 1e-3
    @test mean(abs, pullback_implicit_black_box(v)[2] - v) < 1e-3
end
