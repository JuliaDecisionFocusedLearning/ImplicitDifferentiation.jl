using Pkg
Pkg.activate(@__DIR__)

using AbstractDifferentiation: ForwardDiffBackend
using BenchmarkTools
using CSV
using DataFrames
using ForwardDiff: ForwardDiff, Dual
using ImplicitDifferentiation
using Plots
using Random
using SimpleUnPack
using Zygote: Zygote

forward(x) = sqrt.(x)
conditions(x, y) = abs2.(y) .- x

forward_sum(x) = [sqrt(sum(x))]
conditions_sum(x, y) = [abs2(only(y)) - sum(x)]

forward_fill(x, output_size) = fill(sqrt(only(x)), output_size...)
conditions_fill(x, y, output_size) = abs2.(y) .- only(x)

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
)
    linear_solver = get_linear_solver(linear_solver_symbol)
    conditions_backend = get_conditions_backend(conditions_backend_symbol)

    if scenario_symbol == :jacobian && prod(input_size) >= 1000
        return nothing
    end

    x, implicit = nothing, nothing
    if scenario_symbol == :jacobian
        x = rand(input_size...)
        implicit = ImplicitFunction(forward, conditions; linear_solver, conditions_backend)
    elseif scenario_symbol == :pullback
        x = rand(input_size...)
        implicit = ImplicitFunction(
            forward_sum, conditions_sum; linear_solver, conditions_backend
        )
    elseif scenario_symbol == :pushforward
        @show input_size
        x = Array{Float64,length(input_size)}(undef, ones(Int, length(input_size))...)
        x[only(eachindex(x))] = rand()
        implicit = ImplicitFunction(
            x -> forward_fill(x, input_size),
            (x, y) -> conditions_fill(x, y, input_size);
            linear_solver,
            conditions_backend,
        )
    end

    x_and_dx = Dual.(x, (one(eltype(x)),))
    y = implicit(x)
    dy = zero(y)

    if scenario_symbol == :jacobian && backend_symbol == :ForwardDiff
        return @benchmarkable ForwardDiff.jacobian($implicit, $x) seconds = 1
    elseif scenario_symbol == :jacobian && backend_symbol == :Zygote
        return @benchmarkable Zygote.jacobian($implicit, $x) seconds = 1
    elseif scenario_symbol == :pullback && backend_symbol == :Zygote
        return @benchmarkable begin
            _, back = Zygote.pullback($implicit, $x)
            back($dy)
        end seconds = 1
    elseif scenario_symbol == :pushforward && backend_symbol == :ForwardDiff
        return @benchmarkable $implicit($x_and_dx) seconds = 1
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
)
    SUITE = BenchmarkGroup()

    for sc in scenario_symbols,
        ls in linear_solver_symbols,
        ba in backend_symbols,
        cb in conditions_backend_symbols,
        is in input_sizes

        bench = create_benchmarkable(;
            scenario_symbol=sc,
            linear_solver_symbol=ls,
            backend_symbol=ba,
            conditions_backend_symbol=cb,
            input_size=is,
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
        SUITE[sc][ls][ba][cb][is] = bench
    end
    return SUITE
end

function export_results(
    results;
    scenario_symbols,
    linear_solver_symbols,
    backend_symbols,
    conditions_backend_symbols,
    input_sizes,
    path=joinpath(@__DIR__, "benchmark_results.csv"),
)
    min_results = minimum(results)

    data = DataFrame()

    for sc in scenario_symbols,
        ls in linear_solver_symbols,
        ba in backend_symbols,
        cb in conditions_backend_symbols,
        is in input_sizes

        try
            perf = min_results[sc][ls][ba][cb][is]
            @unpack time, gctime, memory, allocs = perf
            row = (;
                scenario=sc,
                linear_solver=ls,
                backend=ba,
                conditions_backend=cb,
                input_size=is,
                time,
                gctime,
                memory,
                allocs,
            )
            push!(data, row)
        catch KeyError
            nothing
        end
    end

    if !isnothing(path)
        open(path, "w") do file
            CSV.write(file, data)
        end
    end
    return data
end

function plot_results(
    data;
    scenario_symbols,
    linear_solver_symbols,
    backend_symbols,
    conditions_backend_symbols,
    path=joinpath(
        @__DIR__,
        "benchmark_plot_$(scenario_symbols)_$(linear_solver_symbols)_$(backend_symbols)_$(conditions_backend_symbols).pdf",
    ),
)
    pl = plot(;
        size=(1000, 500),
        xlabel="Input dimension (log scale)",
        ylabel="Time [s] (log scale)",
        title="ImplicitDifferentiation.jl benchmarks",
        legendtitle="scen - linsol - back - condback",
        legend=:outerright,
        xaxis=:log10,
        yaxis=:log10,
        margin=5Plots.mm,
    )
    for sc in scenario_symbols,
        ls in linear_solver_symbols,
        ba in backend_symbols,
        cb in conditions_backend_symbols

        filtered_data = subset(
            data,
            :scenario => _col -> _col .== sc,
            :linear_solver => _col -> _col .== ls,
            :backend => _col -> _col .== ba,
            :conditions_backend => _col -> _col .== cb,
        )

        if !isempty(filtered_data)
            x = map(prod, filtered_data[!, :input_size])
            y = filtered_data[!, :time] ./ 1e9
            plot!(
                pl, x, y; linestyle=:auto, markershape=:auto, label="$sc - $ls - $ba - $cb"
            )
        end
    end

    if !isnothing(path)
        savefig(pl, path)
    end
    return pl
end

scenario_symbols = (:jacobian, :pullback, :pushforward)
linear_solver_symbols = (:direct, :iterative)
backend_symbols = (:Zygote, :ForwardDiff)
conditions_backend_symbols = (:nothing, :ForwardDiff)
input_sizes = [(n,) for n in (1, 10, 100, 1_000, 10_000)]

SUITE = make_suite(;
    scenario_symbols,
    linear_solver_symbols,
    backend_symbols,
    conditions_backend_symbols,
    input_sizes,
)

results = run(SUITE; verbose=true, evals=1)

# data = export_results(
#     results;
#     scenario_symbols,
#     linear_solver_symbols,
#     backend_symbols,
#     conditions_backend_symbols,
#     input_sizes,
# )

# plot_results(
#     data;
#     scenario_symbols=[:pullback],
#     linear_solver_symbols=[:direct, :iterative],
#     backend_symbols=[:FowardDiff, :Zygote],
#     conditions_backend_symbols=[:nothing],
# )
