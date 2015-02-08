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

  A = sprand(n,n,0.2)+1im*sprand(n,n,0.2)
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
  e1 = norm(expm(pi/4*[0 1; 1 0])*[1.;0.]-expmv(pi/4,[0 1; 1 0]|>sparse, [1.;0.]));
  e2 = norm(expm(-pi/4*1im*[0 1; 1 0])*[1.;0.]-expmv(pi/4,-1im*[0 1; 1 0]|>sparse, [1.+0im;0.]))
  return e1, e2
end

println("testing n=100 (first expmv, then expm)")
res, t1, t2 = test_expmv(100)
println("residuum: $res\n")
@test res < 1e-6

println("testing complex n=100 (first expmv, then expm)")
res, t1, t2 = test_expmv2(100)
println("residuum: $res\n")
@test res < 1e-6

println("testing n=1000 (first expmv, then expm)")
res, t1, t2 = test_expmv(1000)
println("residuum: $res\n")
@test res < 1e-6

println("testing n=1000 (first expmv, then expm)")
res, t1, t2 = test_expmv2(1000)
println("residuum: $res\n")
@test res < 1e-6

println("testing 2x2 cases")
e1, e2 = test_expmv3()
println("residua: $e1, $e2\n")
@test e1 < 1e-10
@test e2 < 1e-10