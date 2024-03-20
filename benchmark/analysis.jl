
## Benchmark analysis

using CSV
using DataFrames
using Plots

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
