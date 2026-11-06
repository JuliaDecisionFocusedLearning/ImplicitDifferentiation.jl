@testitem "Enzyme mutation after call" begin
    using ImplicitDifferentiation
    using Enzyme
    using Test

    solver(x, a) = a .* sqrt.(x), nothing
    conditions(x, y, z, a) = y .^ 2 .- a .^ 2 .* x
    const implicit = ImplicitFunction(solver, conditions)

    # the arguments are mutated between the forward and reverse passes
    function f_mutate_x(x, a)
        s = sum(first(implicit(x, a)))
        x .= 100
        return s
    end
    function f_mutate_a(x, a)
        s = sum(first(implicit(x, a)))
        a .= 100
        return s
    end
    function f_mutate_y(x, a)
        y = first(implicit(x, a))
        s = sum(y)
        y .= 100
        return s
    end

    x = [1.0, 4.0]
    a = [1.0, 2.0]
    g_true = a ./ (2 .* sqrt.(x))
    for f in (f_mutate_x, f_mutate_a, f_mutate_y)
        dx = zero(x)
        Enzyme.autodiff(Reverse, f, Active, Duplicated(copy(x), dx), Const(copy(a)))
        @test dx ≈ g_true
    end
end
