#=
This file defines a benchmark suite with the tools provided by
`PkgBenchmark` and `BenchmarkTools`.

To run the benchmarks, execute:

```julia
julia> using PkgBenchmark
julia> results = benchmarkpkg("Expokit")
```

To compare current version to another tagged version, commit or branch:

```julia
julia> results = judge("Expokit", <tagged-version-or-branch>)
```

To export the benchmark results to a Markdown file:

```julia
julia> export_markdown("results.md", results)
```

To export the benchmark results to a JSON file:

```julia
julia> writeresults("results.json", results)
```
=#
using BenchmarkTools, Expokit

SUITE = BenchmarkGroup()  # parent BenchmarkGroup to contain our suite

include("slicot/slicot.jl")

# From BenchmarkTools.jl
# ----------------------
# If a cache of tuned parameters already exists, use it, otherwise, tune and cache
# the benchmark parameters. Reusing cached parameters is faster and more reliable
# than re-tuning `suite` every time the file is included.
paramspath = joinpath(dirname(@__FILE__), "params.json")

if isfile(paramspath)
    loadparams!(SUITE, BenchmarkTools.load(paramspath)[1], :evals);
else
    tune!(SUITE)
    BenchmarkTools.save(paramspath, params(SUITE));
end
