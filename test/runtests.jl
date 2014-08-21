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

println("testing n=100 (first expmv, then expm)")
res, t1, t2 = test_expmv(100)
println("residuum: $res\n")
@test res < 1e-6

println("testing n=1000 (first expmv, then expm)")
res, t1, t2 = test_expmv(1000)
println("residuum: $res\n")
@test res < 1e-6
