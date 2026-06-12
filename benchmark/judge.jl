# Compares two saved benchmark runs leaf by leaf. From the repository root:
#
#     julia --project=benchmark benchmark/judge.jl <after.json> <baseline.json>
#
# Each leaf's minimum time is classified as a regression, an improvement, or
# invariant under BenchmarkTools' default 5% time tolerance. Leaves present
# in only one of the two runs are not compared.

using BenchmarkTools

length(ARGS) == 2 || error("usage: judge.jl <after.json> <baseline.json>")

after = BenchmarkTools.load(ARGS[1])[1]
baseline = BenchmarkTools.load(ARGS[2])[1]

comparison = judge(minimum(after), minimum(baseline))

verdicts = BenchmarkTools.leaves(comparison)
for verdict in (:regression, :improvement, :invariant)
    selected = [(path, j) for (path, j) in verdicts if time(j) == verdict]
    isempty(selected) && continue
    println(uppercase(string(verdict)), " (", length(selected), "):")
    for (path, j) in sort(selected; by = first)
        println("  ", join(path, "/"), ": ", j)
    end
    println()
end
