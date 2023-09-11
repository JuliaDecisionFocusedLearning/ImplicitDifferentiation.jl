using BenchmarkTools
using PkgBenchmark

pkg = dirname(@__DIR__) # this git repo
baseline = "112d549"  # commit id
target = "82242b9"  # commit id

results_baseline = benchmarkpkg(pkg, baseline; verbose=true, retune=false)
results_target = benchmarkpkg(pkg, target; verbose=true, retune=false)

judgement = judge(results_target, results_baseline, minimum)

export_markdown(joinpath(@__DIR__, "benchmark_judgement.md"), judgement)
