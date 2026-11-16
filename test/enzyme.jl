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

@testitem "Enzyme type-unstable solver" begin
    using ImplicitDifferentiation
    using Enzyme
    using EnzymeTestUtils
    using Test

    # the output of the solver is not inferred (non-regression test for #234)
    const BOX = Ref{Any}()
    function solver(x, a)
        BOX[] = a .* sqrt.(x)
        return BOX[], nothing
    end
    conditions(x, y, z, a) = y .^ 2 .- a .^ 2 .* x
    const implicit = ImplicitFunction(solver, conditions)
    @test !isconcretetype(Base.promote_op(implicit, Vector{Float64}, Vector{Float64}))

    f(x, a) = sum(first(implicit(x, a)))
    x = [1.0, 4.0]
    a = [1.0, 2.0]
    g_true = a ./ (2 .* sqrt.(x))

    @testset "Reverse" begin
        dx = zero(x)
        Enzyme.autodiff(Reverse, f, Active, Duplicated(copy(x), dx), Const(a))
        @test dx ≈ g_true
    end
    @testset "Forward" begin
        dirs = ([1.0, 0.0], [0.0, 1.0])
        g = map(dirs) do dx
            only(Enzyme.autodiff(Forward, f, Duplicated(copy(x), dx), Const(a)))
        end
        @test collect(g) ≈ g_true
    end
    # batched modes are not tested: Enzyme itself crashes (LLVM assertion) on batched
    # custom rules whose primal return type is not inferred
    @testset "EnzymeTestUtils" begin
        for Tret in (Const, Duplicated, DuplicatedNoNeed)
            test_forward(implicit, Tret, (x, Duplicated), (a, Const))
        end
        for Tret in (Const, Duplicated)
            test_reverse(implicit, Tret, (x, Duplicated), (a, Const))
        end
        for Tret in (Duplicated, BatchDuplicated)
            test_forward(implicit, Tret, (x, Const), (a, Const))
            test_reverse(implicit, Tret, (x, Const), (a, Const))
        end
    end
end
