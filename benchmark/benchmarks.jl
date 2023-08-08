using AbstractDifferentiation: ForwardDiffBackend
using BenchmarkTools
using CSV
using DataFrames
using ForwardDiff: ForwardDiff
using ImplicitDifferentiation
using Random
using SimpleUnPack
using Zygote: Zygote

forward(x) = sqrt.(abs.(x))
conditions(x, y) = abs2.(y) .- abs.(x)

function make_suite(; linear_solvers, conditions_backends, input_sizes)
    SUITE = BenchmarkGroup()
    for linear_solver in linear_solvers
        ls = string(typeof(linear_solver))
        SUITE[ls] = BenchmarkGroup()
        for conditions_backend in conditions_backends
            implicit = ImplicitFunction(
                forward, conditions; linear_solver, conditions_backend
            )
            cb = string(typeof(conditions_backend))
            SUITE[ls][cb] = BenchmarkGroup()
            for input_size in input_sizes
                x = rand(input_size...)
                is = string(input_size)
                g = BenchmarkGroup()
                g["ForwardDiff"] = @benchmarkable ForwardDiff.jacobian($implicit, $x) seconds =
                    1
                g["Zygote"] = @benchmarkable Zygote.jacobian($implicit, $x) seconds = 1
                SUITE[ls][cb][is] = g
            end
        end
    end
    return SUITE
end

function run_suite(;
    linear_solvers, conditions_backends, input_sizes, path=joinpath(@__DIR__, "results.csv")
)
    SUITE = make_suite(; linear_solvers, conditions_backends, input_sizes)

    results = BenchmarkTools.run(SUITE; verbose=true, evals=1)
    min_results = minimum(results)

    data = DataFrame()
    for linear_solver in linear_solvers,
        conditions_backend in conditions_backends,
        input_size in input_sizes,
        backend in ["ForwardDiff", "Zygote"]

        ls = string(typeof(linear_solver))
        cb = string(typeof(conditions_backend))
        is = string(input_size)
        perf = min_results[ls][cb][is][backend]
        @unpack time, gctime, memory, allocs = perf
        row = (;
            func="sqrt",
            linear_solver=ls,
            backend=backend,
            conditions_backend=cb,
            input_size=is,
            time,
            gctime,
            memory,
            allocs,
        )
        push!(data, row)
    end

    if !isnothing(path)
        open(path, "w") do file
            CSV.write(file, data)
        end
    end
    return data
end

linear_solvers = (DirectLinearSolver(), IterativeLinearSolver())
conditions_backends = (nothing, ForwardDiffBackend())
input_sizes = vcat(
    [(n,) for n in (10, 100, 1_000, 10_000)], #
    [(n, n) for n in (10, 30, 90)], #
)

SUITE = make_suite(; linear_solvers, conditions_backends, input_sizes)

# data = run_suite(; linear_solvers, conditions_backends, input_sizes)
