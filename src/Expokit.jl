module Expokit

export expv, expv!


const axpy! = Base.LinAlg.axpy!
const expm! = Base.LinAlg.expm!

###############################################################################
# calculate matrix exponential acting on some vector, w = exp(t*A)*v,
# using the Krylov subspace approximation
#
# see R.B. Sidje, ACM Trans. Math. Softw., 24(1):130-156, 1998
# and http://www.maths.uq.edu.au/expokit
#
#
expv{T}( vec::Vector{T}, t::Real, amat::AbstractMatrix;
				 tol::Real=1e-7, m::Int=min(30,size(amat,1)), norm=Base.norm) = expv!(copy(vec), t, amat; tol=tol, m=m, norm=norm)

function expv!{T}( vec::Vector{T}, t::Real, amat::AbstractMatrix;
                    tol::Real=1e-7, m::Int=min(30,size(amat,1)), norm=Base.norm)

  if size(vec,1) != size(amat,2)
    error("dimension mismatch")
  end

  # safety factors
  gamma = 0.9
  delta = 1.2

  btol = 1e-7 	# tolerance for "happy-breakdown"
  maxiter = 10	# max number of time-step refinements

  anorm = norm(amat, Inf)
  rndoff= anorm*eps()

  # estimate first time-step and round to two significant digits
  beta = norm(vec)
  r = 1/m
  fact = (((m+1)/exp(1.))^(m+1))*sqrt(2.*pi*(m+1))
  tau = (1./anorm)*((fact*tol)/(4.*beta*anorm))^r
  tau = signif(tau, 2)

  vm = Array(Vector{T}, m+1)
  for i=1:m+1
    vm[i]=similar(vec)
  end
  hm = zeros(T,m+2,m+2)

  tf = abs(t)
  tsgn = sign(t)
  tk = zero(t)

  v = vec
  p = similar(v)
  mx = m
  while tk < tf
    tau = min(tf-tk, tau)

    # Arnoldi procedure
    # vm[1] = v/beta
    scale!(copy!(vm[1],v),1/beta)
    mx = m
    for j=1:m
      # p[:] = amat*vm[j]
      Base.A_mul_B!(p, amat, vm[j])

      for i=1:j
        hm[i,j] = dot(vm[i], p)
        # p[:] = p - hm[i,j]*vm[i]
        p = axpy!(-hm[i,j], vm[i], p)
      end
      hm[j+1,j] = norm(p)

      if real(hm[j+1,j]) < btol	# happy-breakdown
        tau = tf - tk
        err_loc = btol

        # F = expm(tsgn*tau*hm[1:j,1:j])
        F = expm!(scale!(tsgn*tau,slice(hm,1:j,1:j)))
        fill!(v, zero(T))
        for k=1:j
          # v[:] = v + beta*vm[k]*F[k,1]
          v = axpy!(beta*F[k,1], vm[k], v)
        end
        mx = j
        break
      end

      # vm[j+1] = p/hm[j+1,j]
      scale!(copy!(vm[j+1],p),1/hm[j+1,j])
    end
    hm[m+2,m+1] = one(T)
    (mx != m) || (avnorm = norm(Base.A_mul_B!(p,amat,vm[m+1])))

    # propagate using adaptive step size
    iter = 1
    while (iter < maxiter) && (mx == m)

      # F = expm(tsgn*tau*hm)
      F = expm!(scale(tsgn*tau,hm))

      # local error estimation
      err1 = abs( beta*F[m+1,1] )
      err2 = abs( beta*F[m+2,1] * avnorm )

      if err1 > 10*err2	# err1 >> err2
        err_loc = err2
        r = 1/m
      elseif err1 > err2
        err_loc = (err1*err2)/(err1-err2)
        r = 1/m
      else
        err_loc = err1
        r = 1/(m-1)
      end

      # time-step sufficient?
      if err_loc <= delta * tau * (tau*tol/err_loc)^r
        fill!(v, zero(T))
        for k=1:m+1
          # v[:] = v + beta*vm[k]*F[k,1]
          v = axpy!(beta*F[k,1], vm[k], v)
        end

        break
      end
      tau = gamma * tau * (tau*tol/err_loc)^r		# estimate new time-step
      tau = signif(tau, 2)				# round to 2 signiﬁcant digits
                                  # to prevent numerical noise
      iter = iter + 1
    end
    if iter == maxiter
      # TODO, here an exception should be thrown, but which?
      error("Number of iterations exceeded $(maxiter). Requested tolerance might be to high.")
    end

    beta = norm(v)
    tk = tk + tau

    tau = gamma * tau * (tau*tol/err_loc)^r		# estimate new time-step
    tau = signif(tau, 2)			# round to 2 signiﬁcant digits
                              # to prevent numerical noise
    err_loc = max(err_loc,rndoff)
  end

  return v
end # expv!

end # module
