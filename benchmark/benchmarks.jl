using Pkg
Pkg.activate(@__DIR__)

using AbstractDifferentiation: ForwardDiffBackend
using BenchmarkTools
using ForwardDiff: ForwardDiff, Dual
using ProgressMeter
using Random
using SimpleUnPack
using Zygote: Zygote

using ImplicitDifferentiation

## Benchmark definition

forward(x; output_size) = fill(sqrt(sum(x)), output_size...)
conditions(x, y; output_size) = abs2.(y) .- sum(x)

function get_linear_solver(linear_solver_symbol::Symbol)
    if linear_solver_symbol == :direct
        return DirectLinearSolver()
    elseif linear_solver_symbol == :iterative
        return IterativeLinearSolver()
    end
end

function get_conditions_backend(conditions_backend_symbol::Symbol)
    if conditions_backend_symbol == :nothing
        return nothing
    elseif conditions_backend_symbol == :ForwardDiff
        return ForwardDiffBackend()
    end
end

function create_benchmarkable(;
    scenario_symbol,
    linear_solver_symbol,
    backend_symbol,
    conditions_backend_symbol,
    input_size,
    output_size,
)
    linear_solver = get_linear_solver(linear_solver_symbol)
    conditions_backend = get_conditions_backend(conditions_backend_symbol)

    if scenario_symbol == :jacobian && prod(input_size) * prod(output_size) >= 10^5
        return nothing
    end

    x = rand(input_size...)
    implicit = ImplicitFunction(
        x -> forward(x; output_size),
        (x, y) -> conditions(x, y; output_size);
        linear_solver,
        conditions_backend,
    )

    dx = similar(x)
    dx .= one(eltype(x))
    x_and_dx = Dual.(x, dx)
    y = implicit(x)
    dy = similar(y)
    dy .= one(eltype(y))

    if scenario_symbol == :jacobian && backend_symbol == :ForwardDiff
        return @benchmarkable ForwardDiff.jacobian($implicit, $x) seconds = 1 samples = 100
    elseif scenario_symbol == :jacobian && backend_symbol == :Zygote
        return @benchmarkable Zygote.jacobian($implicit, $x) seconds = 1 samples = 100
    elseif scenario_symbol == :rrule && backend_symbol == :Zygote
        return @benchmarkable Zygote.pullback($implicit, $x) seconds = 1 samples = 100
    elseif scenario_symbol == :pullback && backend_symbol == :Zygote
        _, back = Zygote.pullback(implicit, x)
        return @benchmarkable ($back)($dy) seconds = 1 samples = 100
    elseif scenario_symbol == :pushforward && backend_symbol == :ForwardDiff
        return @benchmarkable $implicit($x_and_dx) seconds = 1 samples = 100
    else
        return nothing
    end
end

function make_suite(;
    scenario_symbols,
    linear_solver_symbols,
    backend_symbols,
    conditions_backend_symbols,
    input_sizes,
    output_sizes,
)
    SUITE = BenchmarkGroup()

    for sc in scenario_symbols,
        ls in linear_solver_symbols,
        ba in backend_symbols,
        cb in conditions_backend_symbols,
        is in input_sizes,
        os in output_sizes

        bench = create_benchmarkable(;
            scenario_symbol=sc,
            linear_solver_symbol=ls,
            backend_symbol=ba,
            conditions_backend_symbol=cb,
            input_size=is,
            output_size=os,
        )

        isnothing(bench) && continue

        if !haskey(SUITE, sc)
            SUITE[sc] = BenchmarkGroup()
        end
        if !haskey(SUITE[sc], ls)
            SUITE[sc][ls] = BenchmarkGroup()
        end
        if !haskey(SUITE[sc][ls], ba)
            SUITE[sc][ls][ba] = BenchmarkGroup()
        end
        if !haskey(SUITE[sc][ls][ba], cb)
            SUITE[sc][ls][ba][cb] = BenchmarkGroup()
        end
        if !haskey(SUITE[sc][ls][ba][cb], is)
            SUITE[sc][ls][ba][cb][is] = BenchmarkGroup()
        end
        SUITE[sc][ls][ba][cb][is][os] = bench
    end
    return SUITE
end

scenario_symbols = (:jacobian, :rrule, :pullback, :pushforward)
linear_solver_symbols = (:direct, :iterative)
backend_symbols = (:Zygote, :ForwardDiff)
conditions_backend_symbols = (:nothing, :ForwardDiff)
input_sizes = [(n,) for n in floor.(Int, 10 .^ (0:1:3))];
output_sizes = [(n,) for n in floor.(Int, 10 .^ (0:1:3))];

SUITE = make_suite(;
    scenario_symbols,
    linear_solver_symbols,
    backend_symbols,
    conditions_backend_symbols,
    input_sizes,
    output_sizes,
)
