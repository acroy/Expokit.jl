using JLD

S = SUITE["SLICOT"] = BenchmarkGroup(["expmv"])

begin
    slicot_models = load("benchmark/slicot/slicot.jld")
    for si in keys(slicot_models)
        A = slicot_models[si]["A"]
        N = eltype(A)
        n = size(A, 1)

        c =  zeros(N, n) # concentrated weight
        c[1] = one(N)
        S[si, n, "c"] = @benchmarkable expmv(-1e-3, $A, $c) 

        u = fill(one(N)/sqrt(n), n)
        S[si, n, "u"] = @benchmarkable expmv(-1e-3, $A, $u) # uniform weight
    end
end
