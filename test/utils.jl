using ADTypes
using ADTypes: ForwardMode, ReverseMode, ForwardOrReverseMode
using ChainRulesCore
using ChainRulesTestUtils
using ComponentArrays
import DifferentiationInterface as DI
using ForwardDiff: ForwardDiff
import ImplicitDifferentiation as ID
using ImplicitDifferentiation: ImplicitFunction, prepare_implicit
using JET
using LinearAlgebra
using Random: rand!
using Test
using Zygote: Zygote, ZygoteRuleConfig

@kwdef struct Scenario{S,C,X,A,K,Xp,Ap}
    solver::S
    conditions::C
    x::X
    args::A = ()
    implicit_kwargs::K = (;)
    x_prep::Xp = zero(x)
    args_prep::Ap = map(zero, args)
end

function Base.show(io::IO, scen::Scenario)
    print(
        io,
        "Scenario(; solver=$(scen.solver), conditions=$(scen.conditions), x::$(typeof(scen.x))",
    )
    if !isempty(scen.args)
        print(io, ", args::$(typeof(scen.args))")
    end
    if !isempty(scen.implicit_kwargs)
        print(io, ", implicit_kwargs::$(typeof(scen.implicit_kwargs))")
    end
    return print(io, ")")
end

function identity_break_autodiff(x::AbstractArray{R}) where {R}
    float(first(x))  # break ForwardDiff
    (Vector{R}(undef, 1))[1] = first(x)  # break Zygote
    result = try
        throw(copy(x))
    catch y
        y  # presumably break Enzyme
    end
    return result
end

struct NonDifferentiable{S}
    solver::S
end

(nd::NonDifferentiable)(x, args...) = nd.solver(identity_break_autodiff(x), args...)

function add_arg_mult(scen::Scenario, a=3)
    @assert isempty(scen.args)
    function solver_with_arg_mult(x, a)
        y, z = scen.solver(x)
        return y .* a, z
    end
    function conditions_with_arg_mult(x, y, z, a)
        return scen.conditions(x, y ./ a, z)
    end
    implicit_kwargs_with_arg_mult = NamedTuple(
        Dict(k => if k == :input_example
            (only(v), a)
        else
            v
        end for (k, v) in pairs(scen.implicit_kwargs))
    )

    return Scenario(;
        solver=solver_with_arg_mult,
        conditions=conditions_with_arg_mult,
        x=scen.x,
        args=(a,),
        implicit_kwargs=implicit_kwargs_with_arg_mult,
        x_prep=scen.x_prep,
        args_prep=(zero(a),),
    )
end

function test_implicit_call(scen::Scenario)
    implicit = ImplicitFunction(
        NonDifferentiable(scen.solver), scen.conditions; scen.implicit_kwargs...
    )
    y, z = implicit(scen.x, scen.args...)
    y_true, z_true = scen.solver(scen.x, scen.args...)

    @testset "Call" begin
        @test y ≈ y_true
        @test z == z_true
    end
end

tag(::AbstractArray{<:ForwardDiff.Dual{T}}) where {T} = T

function test_implicit_duals(scen::Scenario; type_stability::Bool)
    implicit = ImplicitFunction(
        NonDifferentiable(scen.solver), scen.conditions; scen.implicit_kwargs...
    )
    prep = prepare_implicit(ForwardMode(), implicit, scen.x_prep, scen.args_prep...)

    dx = similar(scen.x)
    rand!(dx)
    x_and_dx = ForwardDiff.Dual.(scen.x, dx)

    y_true, z_true = scen.solver(scen.x, scen.args...)
    dy_true = DI.pushforward(
        first ∘ scen.solver,
        AutoForwardDiff(),
        scen.x,
        (dx,),
        map(DI.Constant, scen.args)...,
    )[1]

    @testset "Duals" begin
        @testset "Prepared" begin
            y_and_dy, z = implicit(prep, x_and_dx, scen.args...)
            T = tag(y_and_dy)
            y = ForwardDiff.value.(y_and_dy)
            dy = ForwardDiff.extract_derivative.(T, y_and_dy)
            @test y ≈ y_true
            @test dy ≈ dy_true
            @test z == z_true
            if type_stability
                @inferred implicit(prep, x_and_dx, scen.args...)
            end
        end
        @testset "Unrepared" begin
            y_and_dy, z = implicit(x_and_dx, scen.args...)
            T = tag(y_and_dy)
            y = ForwardDiff.value.(y_and_dy)
            dy = ForwardDiff.extract_derivative.(T, y_and_dy)
            @test y ≈ y_true
            @test dy ≈ dy_true
            @test z == z_true
            if type_stability
                @inferred implicit(x_and_dx, scen.args...)
            end
        end
    end
end

function test_implicit_rrule(scen::Scenario; type_stability::Bool)
    implicit = ImplicitFunction(
        NonDifferentiable(scen.solver), scen.conditions; scen.implicit_kwargs...
    )
    y_true, z_true = scen.solver(scen.x, scen.args...)

    dy = similar(y_true)
    rand!(dy)
    dz = NoTangent()

    dx_true = DI.pullback(
        first ∘ scen.solver, AutoZygote(), scen.x, (dy,), map(DI.Constant, scen.args)...
    )[1]

    @testset "ChainRule" begin
        (y, z), pb = rrule_via_ad(ZygoteRuleConfig(), implicit, scen.x, scen.args...)
        dimpl, dx = pb((dy, dz))
        @test y ≈ y_true
        @test z == z_true
        @test dimpl isa NoTangent
        @test dx ≈ dx_true
        if type_stability
            @inferred rrule_via_ad(ZygoteRuleConfig(), implicit, scen.x, scen.args...)
            @inferred pb((dy, dz))
        end
    end
end

function test_implicit_jacobian(scen::Scenario, outer_backend::AbstractADType)
    implicit = ImplicitFunction(
        NonDifferentiable(scen.solver), scen.conditions; scen.implicit_kwargs...
    )
    prep = prepare_implicit(
        ForwardOrReverseMode(), implicit, scen.x_prep, scen.args_prep...
    )
    jac_true = DI.jacobian(
        first ∘ scen.solver, outer_backend, scen.x, map(DI.Constant, scen.args)...
    )

    @testset "Jacobian - $outer_backend" begin
        if outer_backend isa AutoForwardDiff
            @testset "Prepared" begin
                jac = DI.jacobian(
                    x -> first(implicit(prep, x, scen.args...)), outer_backend, scen.x
                )
                @test jac ≈ jac_true
            end
        end
        @testset "Unprepared" begin
            jac = DI.jacobian(
                first ∘ implicit, outer_backend, scen.x, map(DI.Constant, scen.args)...
            )
            @test jac ≈ jac_true
        end
    end
end

function test_implicit(
    scen::Scenario,
    outer_backends=[AutoForwardDiff(), AutoZygote()];
    type_stability::Bool=false,
)
    return @testset "$scen" begin
        test_implicit_call(scen)
        test_implicit_duals(scen; type_stability)
        test_implicit_rrule(scen; type_stability)
        for outer_backend in outer_backends
            test_implicit_jacobian(scen, outer_backend)
        end
    end
end

default_solver(x) = vcat(sqrt.(x .+ 2), -sqrt.(x)), 2
default_conditions(x, y, z) = abs2.(y) .- vcat(x .+ z, x)
