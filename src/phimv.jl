export phimv, phimv!

"""
    phimv{T}(t, A, u, vec; [tol], [m], [norm], [anorm])

Calculate the solution of a nonhomogeneous linear ODE problem with constant input
``w = e^{tA}v + tφ(tA)u`` using the Krylov subspace approximation.

# Input

- `A`   -- matrix which can be dense or sparse
- `u`   -- vector, constant input of the ODE
- `vec` -- vector on which the matrix exponential of ``A`` is applied
- `tol` -- (optional, default: 1e-7) the requested accuracy tolerance on ``w``
- `m`   -- (optional, default: min(30, n)) maximum size of the Krylov basis

---

    [w, err] = phiv( t, A, u, v, tol, m )
    PHIV computes an approximation of ``w = exp(tA)v + t φ(tA)u``
    for a general matrix A using Krylov subspace projection techniques.
    Here, ``φ(z) = (\\exp(z)-1)/z`` and ``w`` is the solution of the
    nonhomogeneous linear ODE problem ``w' = Aw + u``, ``w(0) = v``.
    It does not compute the matrix functions in isolation but instead,
    it computes directly the action of these functions on the
    operand vectors. This way of doing so allows for addressing large
    sparse problems. The matrix under consideration interacts only
    via matrix-vector products (matrix-free method).

    Roger B. Sidje (rbs@maths.uq.edu.au)
    EXPOKIT: Software Package for Computing Matrix Exponentials.
    ACM - Transactions On Mathematical Software, 24(1):130-156, 1998
"""
function phimv( t::Number,
                   A, u::Vector{T}, vec::Vector{T};
                   tol::Real=1e-7,
                   m::Int=min(30, size(A, 1)),
                   norm=LinearAlgebra.norm, anorm=default_anorm(A)) where {T}

    result = convert(Vector{promote_type(eltype(A), T, typeof(t))}, copy(vec))
    phimv!(t, A, u, result; tol=tol, m=m, norm=norm, anorm=anorm)
    return result
end

phimv!( t::Number,
           A, u::Vector{T}, vec::Vector{T};
           tol::Real=1e-7,
           m::Int=min(30, size(A, 1)),
           norm=LinearAlgebra.norm, anorm=default_anorm(A)) where {T} = phimv!(vec, t, A, u, vec; tol=tol, m=m, norm=norm, anorm=anorm)

function phimv!( w::Vector{T}, t::Number, A, u::Vector{T}, vec::Vector{T};
                   tol::Real=1e-7, m::Int=min(30, size(A, 1)), norm=LinearAlgebra.norm, anorm=default_anorm(A)) where {T}

    if size(vec, 1) != size(A, 2)
        error("dimension mismatch")
    end

    # safety factors
    gamma = 0.9
    delta = 1.2

    btol = 1e-7     # tolerance for "happy-breakdown"
    maxiter = 10    # max number of time-step refinements

    rndoff = anorm*eps()

    # estimate first time-step and round to two significant digits
    beta = norm(A*vec + u)
    r = 1/m
    fact = (((m+1)/exp(1))^(m+1))*sqrt(2*pi*(m+1))
    tau = (1.0/anorm)*((fact*tol)/(4.0*beta*anorm))^r
    tau = round(tau, sigdigits=2)

    # storage for Krylov subspace vectors
    vm = Array{typeof(w)}(undef,m+1)
    for i = 1:m+1
        vm[i] = similar(w)
    end
    hm = zeros(T, m+3, m+3)

    tf = abs(t)
    tsgn = sign(t)
    tk = zero(tf)

    copyto!(w, vec)
    p = similar(w)
    k1 = 3
    mb = m

    while tk < tf
        tau = min(tf-tk, tau)

        # Arnoldi procedure
        rmul!(copyto!(vm[1], A*w+u), 1/beta)

        for j = 1:m
            mul!(p, A, vm[j])

            for i = 1:j
                hm[i, j] = dot(vm[i], p)
                p = axpy!(-hm[i,j], vm[i], p)
            end
            s = norm(p)

            if s < btol # happy-breakdown
                tau = tf - tk
                k1 = 0
                mb = j
                break
            end
            hm[j+1,j] = s
            rmul!(copyto!(vm[j+1], p), 1/hm[j+1,j])
        end

        hm[1, mb+1] = one(T)
        if k1 != 0
            hm[m+1,m+2] = one(T); hm[m+2,m+3] = one(T)
            h = hm[m+1,m]; hm[m+1,m] = zero(T)
            avnorm = norm(A*vm[m+1])
        end

        local F
        # propagate using adaptive step size
        iter = 0
        while iter <= maxiter
            mx = mb + max(1,k1)
            F = exp!(tsgn*tau*view(hm, 1:mx, 1:mx))
            if k1 == 0
                err_loc = btol
                break
            else
                F[m+1,m+1] = h*F[m,m+2]
                F[m+2,m+1] = h*F[m,m+3]
                err1 = abs(beta*F[m+1,m+1])
                err2 = abs(beta*F[m+2,m+1] * avnorm)
                if err1 > 10*err2
                    err_loc = err2
                    r = 1/m
                elseif err1 > err2
                    err_loc = (err1*err2)/(err1-err2)
                    r = 1/m
                else
                    err_loc = err1
                    r = 1/(m-1)
                end
            end

            # time-step sufficient?
            if err_loc <= delta * tau*tol
                break
            else
                tau = gamma * tau * (tau*tol/err_loc)^r # estimate new time-step
                tau = round(tau, sigdigits=2) # round to 2 signiﬁcant digits
                                     # to prevent numerical noise
                if iter == maxiter
                    error("Number of iterations exceeded $(maxiter). Requested tolerance might be too high.")
                end
                iter = iter + 1
            end
        end
        mx = mb + max(0, k1-2)

        for k = 1:mx
            w = axpy!(beta*F[k, mb+1], vm[k], w)
        end

        beta = norm(A*w + u)
        tk = tk + tau

        tau = gamma * tau * (tau*tol/err_loc)^r # estimate new time-step
        tau = round(tau, sigdigits=2) # round to 2 signiﬁcant digits
                             # to prevent numerical noise

        err_loc = max(err_loc, rndoff)

        fill!(hm, zero(T))
    end
    return w
end # phimv!
