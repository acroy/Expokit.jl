using Expokit
using Base.Test

function test_expmv(n::Int)

    A = sprand(n,n,0.4)
    v = eye(n,1)[:]

    tic()
    w1 = expmv(1.0, A, v)
    t1 = toc()

    tic()
    w2 = expm(full(A))*v
    t2 = toc()

    return norm(w1-w2)/norm(w2), t1, t2
end

function test_expmv2(n::Int)

    A = sprand(n,n,0.2) + 1im*sprand(n,n,0.2)
    v = eye(n,1)[:]+0im*eye(n,1)[:]

    tic()
    w1 = expmv(1.0, A, v)
    t1 = toc()

    tic()
    w2 = expm(full(A))*v
    t2 = toc()

    return norm(w1-w2)/norm(w2), t1, t2
end

function test_expmv3()
    e1 = norm(expm(pi/4*[0 1; 1 0])*[1.; 0.] - expmv(pi/4,[0 1; 1 0]|>sparse, [1.; 0.]))
    e2 = norm(expm(-pi/4*1im*[0 1; 1 0])*[1.; 0.] - expmv(pi/4,-1im*[0 1; 1 0]|>sparse, [1.+0im; 0.]))

    return e1, e2
end

println("testing real n=100 (first expmv, then expm)")
res, t1, t2 = test_expmv(100)
println("residuum: $res\n")
@test res < 1e-6

println("testing complex n=100 (first expmv, then expm)")
res, t1, t2 = test_expmv2(100)
println("residuum: $res\n")
@test res < 1e-6

println("testing real n=1000 (first expmv, then expm)")
res, t1, t2 = test_expmv(1000)
println("residuum: $res\n")
@test res < 1e-6

println("testing complex n=1000 (first expmv, then expm)")
res, t1, t2 = test_expmv2(1000)
println("residuum: $res\n")
@test res < 1e-6

println("testing 2x2 cases")
e1, e2 = test_expmv3()
println("residua: $e1, $e2\n")
@test e1 < 1e-10
@test e2 < 1e-10

function test_padm(n::Int)

    A = sprand(n,n,0.001)

    tic()
    w1 = padm(A)
    t1 = toc()

    tic()
    w2 = expm(full(A))
    t2 = toc()

    return norm(w1-w2)/norm(w2), t1, t2
end

function test_padm2(n::Int)

    A = sprand(n,n,0.0005) + 1im*sprand(n,n,0.0005)

    tic()
    w1 = padm(A)
    t1 = toc()

    tic()
    w2 = expm(full(A))
    t2 = toc()

    return norm(w1-w2)/norm(w2), t1, t2
end

println("testing real n=1000 (first padm, then expm)")
res, t1, t2 = test_padm(1000)
println("residuum: $res\n")
@test res < 1e-6

println("testing complex n=100 (first padm, then expm)")
res, t1, t2 = test_padm2(1000)
println("residuum: $res\n")
@test res < 1e-6

function test_chbv(n::Int)

    p = 0.1
    D = diagm(-rand(n))
    T = sprandn(n, n, p)
    H = T * D * T.'  # random negative semidefinite symmetric matrix
    vec = randn(n)
    w1 = chbv(H, vec);

    tic()
    w1 = chbv(H, vec)
    t1 = toc()

    w2 = expm(full(H)) * vec
    tic()
    w2 = expm(full(H)) * vec
    t2 = toc()

    return norm(w1-w2)/norm(w2), t1, t2
end

function test_chbv2(n::Int)

    p = 0.1
    D = diagm(-rand(n))
    T = sprandn(n, n, p) + sprandn(n, n, p)*im
    H = T * D * T'  # random negative semidefinite hermitian matrix
    vec = randn(n) + randn(n)*im
    w1 = chbv(H, vec);

    tic()
    w1 = chbv(H, vec)
    t1 = toc()

    w2 = expm(full(H)) * vec
    tic()
    w2 = expm(full(H)) * vec
    t2 = toc()

    return norm(w1-w2)/norm(w2), t1, t2
end

println("testing real n=100 (first chbv, then expm)")
res, t1, t2 = test_chbv(100)
println("residuum: $res\n")
@test res < 1e-6

println("testing complex n=100 (first chbv, then expm)")
res, t1, t2 = test_chbv2(100)
println("residuum: $res\n")
@test res < 1e-6

println("testing real n=1000 (first chbv, then expm)")
res, t1, t2 = test_chbv(1000)
println("residuum: $res\n")
@test res < 1e-6

println("testing complex n=1000 (first chbv, then expm)")
res, t1, t2 = test_chbv2(1000)
println("residuum: $res\n")
@test res < 1e-6

function test_phimv(n::Int)

    p = 0.1
    found = false
    A = sprand(n, n, p)
    u = rand(n)
    x = similar(u)
    while !found
        try
            copy!(x, A\u)
            found = true
        catch
            A = sprand(n, n, p)
        end
    end
    vec = eye(n, 1)[:]

    w1 = phimv(1.0, A, u, vec) # warmup
    tic()
    w1 = phimv(1.0, A, u, vec)
    t1 = toc()

    w2 = expm(full(A))*(vec+x)-x # warmup
    tic()
    w2 = expm(full(A))*(vec+x)-x
    t2 = toc()

    return norm(w1-w2)/norm(w2), t1, t2
end

function test_phimv2(n::Int)

    p = 0.1
    found = false
    A = sprand(n, n, p) + 1im*sprand(n, n, p)
    u = rand(n) + 1im*rand(n)
    x = similar(u)
    while !found
        try
            copy!(x, A\u)
            found = true
        catch
            A = sprand(n, n, p) + 1im*sprand(n, n, p)
        end
    end
    vec = eye(n, 1)[:] + 0im*eye(n,1)[:]

    w1 = phimv(1.0, A, u, vec) # warmup
    tic()
    w1 = phimv(1.0, A, u, vec)
    t1 = toc()

    w2 = expm(full(A))*(vec+x)-x # warmup
    tic()
    w2 = expm(full(A))*(vec+x)-x
    t2 = toc()

    return norm(w1-w2)/norm(w2), t1, t2
end

println("testing real n=100 (first phimv, then expm)")
res, t1, t2 = test_phimv(100)
println("residuum: $res\n")
@test res < 1e-6

println("testing complex n=100 (first phimv, then expm)")
res, t1, t2 = test_phimv2(100)
println("residuum: $res\n")
@test res < 1e-6

println("testing real n=1000 (first phimv, then expm)")
res, t1, t2 = test_phimv(1000)
println("residuum: $res\n")
@test res < 1e-6

println("testing complex n=1000 (first phimv, then expm)")
res, t1, t2 = test_phimv2(1000)
println("residuum: $res\n")
@test res < 1e-6
