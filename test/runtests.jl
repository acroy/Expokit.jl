using Expokit
using Test
using LinearAlgebra
using SparseArrays

# reference matrix exponential
expm_higham(A) = exp(A)

function ev1(n)

    v = zeros(n)
    v[1] = 1
    return v
end 

function test_expmv(n::Int)

    A = sprand(n,n,0.4)
    v = ev1(n)

    t1 = @elapsed w1 = expmv(1.0, A, v)

    t2 = @elapsed w2 = expm_higham(Matrix(A))*v

    return norm(w1-w2)/norm(w2), t1, t2
end

function test_expmv2(n::Int)

    A = sprand(n,n,0.2) + 1im*sprand(n,n,0.2)
    v = ev1(n) + 0im*ev1(n)

    t1 = @elapsed w1 = expmv(1.0, A, v)

    t2 = @elapsed w2 = expm_higham(Matrix(A))*v

    return norm(w1-w2)/norm(w2), t1, t2
end

function test_expmv3()
    e1 = norm(expm_higham(pi/4*[0 1; 1 0])*[1.; 0.] - expmv(pi/4,[0 1; 1 0]|>sparse, [1.; 0.]))
    e2 = norm(expm_higham(-pi/4*1im*[0 1; 1 0])*[1.; 0.] - expmv(pi/4,-1im*[0 1; 1 0]|>sparse, [1.0+0im; 0.]))

    return e1, e2
end


# Testing custom type for A
struct LinearOp
    m
end

LinearAlgebra.mul!(y, lo::LinearOp, x) = mul!(y, lo.m, x)
Base.size(lo::LinearOp, i::Int) = size(lo.m, i)
Base.eltype(lo::LinearOp) = eltype(lo.m)

# needed for phimv
import Base: *
*(lo::LinearOp, v::Vector) = mul!(similar(v), lo, v)


function test_expmv_linop(n::Int)
    A = LinearOp(sprand(n,n,0.2) + 1im*sprand(n,n,0.2))
    v = ev1(n) + 0im*ev1(n)

    t1 = @elapsed w1 = expmv(1.0, A, v, anorm=norm(A.m, Inf))

    t2 = @elapsed w2 = expmv(1.0, A, v)

    t3 = @elapsed w3 = expm_higham(Matrix(A.m))*v

    return max(norm(w1-w2)/norm(w2), norm(w1-w3)/norm(w3)), t1, t2, t3
end

println("testing real n=100 (first expmv, then expm)")
res, t1, t2 = test_expmv(100)
println("residuum: $res")
println("time dif: $(t1-t2)\n")
@test res < 1e-6

println("testing complex n=100 (first expmv, then expm)")
res, t1, t2 = test_expmv2(100)
println("residuum: $res")
println("time dif: $(t1-t2)\n")
@test res < 1e-6

println("testing real n=1000 (first expmv, then expm)")
res, t1, t2 = test_expmv(1000)
println("residuum: $res")
println("time dif: $(t1-t2)\n")
@test res < 1e-6

println("testing complex n=1000 (first expmv, then expm)")
res, t1, t2 = test_expmv2(1000)
println("residuum: $res")
println("time dif: $(t1-t2)\n")
@test res < 1e-6

println("testing linear operator n=1000 (first expmv, then expm)")
res, t1, t2, t3 = test_expmv_linop(1000)
println("residuum: $res")
println("time dif: $(t1-t2)\n")
@test res < 1e-6

println("testing 2x2 cases")
e1, e2 = test_expmv3()
println("residua: $e1, $e2\n")
@test e1 < 1e-10
@test e2 < 1e-10

function test_padm(n::Int)

    A = sprand(n,n,0.001)

    t1 = @elapsed w1 = padm(A)

    t2 = @elapsed w2 = expm_higham(Matrix(A))

    return norm(w1-w2)/norm(w2), t1, t2
end

function test_padm2(n::Int)

    A = sprand(n,n,0.0005) + 1im*sprand(n,n,0.0005)

    t1 = @elapsed w1 = padm(A)

    t2 = @elapsed w2 = expm_higham(Matrix(A))

    return norm(w1-w2)/norm(w2), t1, t2
end

println("testing real n=1000 (first padm, then expm)")
res, t1, t2 = test_padm(1000)
println("residuum: $res")
println("time dif: $(t1-t2)\n")
@test res < 1e-6

println("testing complex n=100 (first padm, then expm)")
res, t1, t2 = test_padm2(1000)
println("residuum: $res")
println("time dif: $(t1-t2)\n")
@test res < 1e-6

function test_chbv(n::Int)

    p = 0.1
    D = Matrix(Diagonal(-rand(n)))
    T = sprandn(n, n, p)
    H = T * D * transpose(T)  # random negative semidefinite symmetric matrix
    
    vec = randn(n)
    w1 = chbv(H, vec)
    t1 = @elapsed w1 = chbv(H, vec)

    w2 = expm_higham(Matrix(H)) * vec
    t2 = @elapsed  w2 = expm_higham(Matrix(H)) * vec

    return norm(w1-w2)/norm(w2), t1, t2
end

function test_chbv2(n::Int)

    p = 0.1
    D = Matrix(Diagonal(-rand(n)))
    T = sprandn(n, n, p) + sprandn(n, n, p)*im
    H = T * D * adjoint(T)  # random negative semidefinite hermitian matrix
    
    vec = randn(n) + randn(n)*im
    w1 = chbv(H, vec)
    t1 = @elapsed w1 = chbv(H, vec)

    w2 = expm_higham(Matrix(H)) * vec
    t2 = @elapsed w2 = expm_higham(Matrix(H)) * vec

    return norm(w1-w2)/norm(w2), t1, t2
end

println("testing real n=100 (first chbv, then expm)")
res, t1, t2 = test_chbv(100)
println("residuum: $res")
println("time dif: $(t1-t2)\n")
@test res < 1e-6

println("testing complex n=100 (first chbv, then expm)")
res, t1, t2 = test_chbv2(100)
println("residuum: $res")
println("time dif: $(t1-t2)\n")
@test res < 1e-6

println("testing real n=1000 (first chbv, then expm)")
res, t1, t2 = test_chbv(1000)
println("residuum: $res")
println("time dif: $(t1-t2)\n")
@test res < 1e-6

#= commented since this test takes sometime difs very long
println("testing complex n=1000 (first chbv, then expm)")
res, t1, t2 = test_chbv2(1000)
println("residuum: $res")
println("time dif: $(t1-t2)\n")
@test res < 1e-6
=#

function test_phimv(n::Int)

    p = 0.1
    found = false
    A = sprand(n, n, p)
    u = rand(n)
    x = similar(u)
    while !found
        try
            copyto!(x, A\u)
            found = true
        catch
            A = sprand(n, n, p)
        end
    end
    vec = ev1(n)

    w1 = phimv(1.0, A, u, vec) # warmup
    t1 = @elapsed w1 = phimv(1.0, A, u, vec)

    w2 = expm_higham(Matrix(A))*(vec+x)-x # warmup
    t2 = @elapsed w2 = expm_higham(Matrix(A))*(vec+x)-x

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
            copyto!(x, A\u)
            found = true
        catch
            A = sprand(n, n, p) + 1im*sprand(n, n, p)
        end
    end
    vec = ev1(n) + 0im*ev1(n)

    w1 = phimv(1.0, A, u, vec) # warmup
    t1 = @elapsed w1 = phimv(1.0, A, u, vec)

    w2 = expm_higham(Matrix(A))*(vec+x)-x # warmup
    t2 = @elapsed w2 = expm_higham(Matrix(A))*(vec+x)-x

    return norm(w1-w2)/norm(w2), t1, t2
end

println("testing real n=100 (first phimv, then expm)")
res, t1, t2 = test_phimv(100)
println("residuum: $res")
println("time dif: $(t1-t2)\n")
@test res < 1e-6

println("testing complex n=100 (first phimv, then expm)")
res, t1, t2 = test_phimv2(100)
println("residuum: $res")
println("time dif: $(t1-t2)\n")
@test res < 1e-6

println("testing real n=1000 (first phimv, then expm)")
res, t1, t2 = test_phimv(1000)
println("residuum: $res")
println("time dif: $(t1-t2)\n")
@test res < 1e-6

println("testing complex n=1000 (first phimv, then expm)")
res, t1, t2 = test_phimv2(1000)
println("residuum: $res")
println("time dif: $(t1-t2)\n")
@test res < 1e-6

function test_phimv_linop(n::Int)
    p = 0.1
    found = false
    A = LinearOp(sprand(n,n,p))
    u = rand(n)
    x = similar(u)
    while !found
        try
            copyto!(x, A.m\u)
            found = true
        catch
            A = LinearOp(sprand(n,n,p))
        end
    end
    vec = ev1(n)

    w1 = phimv(1.0, A, u, vec) # warmup
    t1 = @elapsed w1 = phimv(1.0, A, u, vec)

    w2 = expm_higham(Matrix(A.m))*(vec+x)-x # warmup
    t2 = @elapsed w2 = expm_higham(Matrix(A.m))*(vec+x)-x

    return norm(w1-w2)/norm(w2), t1, t2
end

println("testing real n=1000 (first phimv, then expm), the linear operator version.")
res, t1, t2 = test_phimv_linop(1000)
println("residuum: $res")
println("time dif: $(t1-t2)\n")
@test res < 1e-6
