using Pkg
Pkg.activate(@__DIR__)

using AbstractDifferentiation: ForwardDiffBackend
using BenchmarkTools
using CSV
using DataFrames
using ForwardDiff: ForwardDiff, Dual
using Plots
using Random
using SimpleUnPack
using Zygote: Zygote

using ImplicitDifferentiation

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
        return @benchmarkable ForwardDiff.jacobian($implicit, $x)
    elseif scenario_symbol == :jacobian && backend_symbol == :Zygote
        return @benchmarkable Zygote.jacobian($implicit, $x)
    elseif scenario_symbol == :rrule && backend_symbol == :Zygote
        return @benchmarkable Zygote.pullback($implicit, $x)
    elseif scenario_symbol == :pullback && backend_symbol == :Zygote
        _, back = Zygote.pullback(implicit, x)
        return @benchmarkable ($back)($dy)
    elseif scenario_symbol == :pushforward && backend_symbol == :ForwardDiff
        return @benchmarkable $implicit($x_and_dx)
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

function export_results(
    results;
    scenario_symbols,
    linear_solver_symbols,
    backend_symbols,
    conditions_backend_symbols,
    input_sizes,
    output_sizes,
    path=joinpath(@__DIR__, "benchmark_results.csv"),
)
    min_results = minimum(results)

    data = DataFrame()

    for sc in scenario_symbols,
        ls in linear_solver_symbols,
        ba in backend_symbols,
        cb in conditions_backend_symbols,
        is in input_sizes,
        os in output_sizes

        try
            perf = min_results[sc][ls][ba][cb][is][os]
            @unpack time, gctime, memory, allocs = perf
            row = (;
                scenario=sc,
                linear_solver=ls,
                backend=ba,
                conditions_backend=cb,
                input_size=is,
                output_size=os,
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
    scenario::Symbol,
    linear_solver_symbols=unique(data[!, :linear_solver]),
    backend_symbols=unique(data[!, :backend]),
    conditions_backend_symbols=unique(data[!, :conditions_backend]),
    input_size=nothing,
    output_size=nothing,
    path=joinpath(
        @__DIR__,
        "benchmark_plot_$(scenario)_$(linear_solver_symbols)_$(backend_symbols)_$(conditions_backend_symbols)_$(input_size)_$(output_size).png",
    ),
)
    pl = plot(;
        size=(800, 400),
        ylabel="Time [s] (log)",
        legendtitle="lin. solver / AD / cond. AD",
        legend=:outerright,
        xaxis=:log10,
        yaxis=:log10,
        margin=5Plots.mm,
        legendtitlefontsize=7,
        legendfontsize=6,
    )

    data = subset(data, :scenario => _col -> _col .== scenario)

    if isnothing(input_size) && isnothing(output_size)
        error("Cannot plot if neither input nor output size is fixed")
    elseif !isnothing(input_size) && !isnothing(output_size)
        error("Cannot plot if both input and output size are fixed")
    elseif !isnothing(input_size)
        plot!(
            pl;
            xlabel="Output dimension (log)",
            title="Implicit diff. - $scenario - input size $input_size",
        )
        data = subset(data, :input_size => _col -> _col .== Ref(input_size))
    else
        plot!(
            pl;
            xlabel="Input dimension (log)",
            title="Implicit diff. - $scenario - output size $output_size",
        )
        data = subset(data, :output_size => _col -> _col .== Ref(output_size))
    end

    for ls in linear_solver_symbols, ba in backend_symbols, cb in conditions_backend_symbols
        filtered_data = subset(
            data,
            :linear_solver => _col -> _col .== ls,
            :backend => _col -> _col .== ba,
            :conditions_backend => _col -> _col .== cb,
        )

        if !isempty(filtered_data)
            x = nothing
            if !isnothing(output_size)
                x = map(prod, filtered_data[!, :input_size])
            elseif !isnothing(output_size)
                x = map(prod, filtered_data[!, :output_size])
            end
            y = filtered_data[!, :time] ./ 1e9
            plot!(
                pl,
                x,
                y;
                linestyle=:auto,
                markershape=:auto,
                label="$ls / $ba / $(cb == :nothing ? ba : cb)",
            )
        end
    end

    if !isnothing(path)
        savefig(pl, path)
    end
    return pl
end

scenario_symbols = (:jacobian, :rrule, :pullback, :pushforward)
linear_solver_symbols = (:direct, :iterative)
backend_symbols = (:Zygote, :ForwardDiff)
conditions_backend_symbols = (:nothing, :ForwardDiff)
input_sizes = [(n,) for n in floor.(Int, 10 .^ (0:0.5:3))];
output_sizes = [(n,) for n in floor.(Int, 10 .^ (0:0.5:3))];

SUITE = make_suite(;
    scenario_symbols,
    linear_solver_symbols,
    backend_symbols,
    conditions_backend_symbols,
    input_sizes,
    output_sizes,
)

# results = BenchmarkTools.run(SUITE; verbose=true, evals=1, seconds=1)

# data = export_results(
#     results;
#     scenario_symbols,
#     linear_solver_symbols,
#     backend_symbols,
#     conditions_backend_symbols,
#     input_sizes,
#     output_sizes,
# )

# plot_results(data; scenario=:pullback, input_size=(1,))
# plot_results(data; scenario=:pullback, input_size=(10,))
# plot_results(data; scenario=:pullback, input_size=(100,))
# plot_results(data; scenario=:pullback, input_size=(1000,))

# plot_results(data; scenario=:pushforward, output_size=(1,))
# plot_results(data; scenario=:pushforward, output_size=(10,))
# plot_results(data; scenario=:pushforward, output_size=(100,))
# plot_results(data; scenario=:pushforward, output_size=(1000,))

# plot_results(data; scenario=:rrule, input_size=(1,))
# plot_results(data; scenario=:rrule, input_size=(10,))
# plot_results(data; scenario=:rrule, input_size=(100,))
# plot_results(data; scenario=:rrule, input_size=(1000,))

# plot_results(data; scenario=:jacobian, input_size=(1,))
# plot_results(data; scenario=:jacobian, input_size=(10,))
# plot_results(data; scenario=:jacobian, input_size=(100,))
# plot_results(data; scenario=:jacobian, input_size=(1000,))
