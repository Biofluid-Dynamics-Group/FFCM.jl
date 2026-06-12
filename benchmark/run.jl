# Runs the benchmark suite and saves the results for later comparison by
# judge.jl. From the repository root:
#
#     julia -t auto --project=benchmark benchmark/run.jl [output.json]
#
# The default output path is benchmark/results/<short-sha>.json; the
# directory is git-ignored because timings are hardware-specific. A .info
# sidecar records the session metadata needed to interpret a result later.

using BenchmarkTools
using Dates: now

include(joinpath(@__DIR__, "benchmarks.jl"))

repo = dirname(@__DIR__)
sha = strip(read(`git -C $repo rev-parse --short HEAD`, String))
dirty = !isempty(strip(read(`git -C $repo status --porcelain`, String)))

resultfile = if isempty(ARGS)
    joinpath(@__DIR__, "results", sha * (dirty ? "-dirty" : "") * ".json")
else
    abspath(ARGS[1])
end
mkpath(dirname(resultfile))

results = run(SUITE; verbose = true)

BenchmarkTools.save(resultfile, results)
open(resultfile * ".info", "w") do io
    println(io, "timestamp:     ", now())
    println(io, "commit:        ", sha, dirty ? " (dirty tree)" : "")
    println(io, "julia:         ", VERSION)
    println(io, "julia_threads: ", Threads.nthreads())
    println(io, "cpu:           ", Sys.cpu_info()[1].model)
    println(io, "machine:       ", Sys.MACHINE)
    println(io, "sizes:         ", join(_selected_sizes(), ","))
end

println("\nResults written to ", resultfile)
