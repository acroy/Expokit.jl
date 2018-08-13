export padm

const PADE_COEFFS = convert(Vector{Float64}, [1//1,
                              1//2,
                              8188362958855447//72057594037927936,
                              8734253822779143//576460752303423488,
                              2911417940926381//2305843009213693952,
                              145570897046319//2305843009213692962,
                              2797136075159//1860878688081779609])

"""
    padm(A; p=6)

Calculate matrix exponential using Pade approximants.

`padm` uses the irreducible (p, p)-degree rational Pade approximation to the
exponential function. The result is always a dense matrix.

# Input

- `A` -- matrix which can be dense or sparse
- `p` -- (optional, default: 6) degree of the rational Pade approximation to
         the exponential function

# Notes

This Julia implementation originated from Expokit's PADM Matlab code by
Roger B. Sidje, see below.

---

  E = padm( A, p )
  PADM computes the matrix exponential exp(A) using the irreducible 
  (p,p)-degree rational Pade approximation to the exponential function.

  E = padm( A )
  p is internally set to 6 (recommended and generally satisfactory).

  See also CHBV, EXPOKIT and the MATLAB supplied functions EXPM and EXPM1.

  Roger B. Sidje (rbs@maths.uq.edu.au)
  EXPOKIT: Software Package for Computing Matrix Exponentials.
  ACM - Transactions On Mathematical Software, 24(1):130-156, 1998
"""
function padm(A; p::Int64=6)

    # Pade coefficients
    if p == 6
        c = copy(PADE_COEFFS)
    else
        c = Float64[]
        push!(c, 1.0)
        @inbounds for k = 1:p
            push!(c, c[end] * ((p+1-k)/(k*(2*p+1-k))))
        end
    end

    # scaling
    normA = opnorm(A, Inf)
    s = 0
    if normA > 0.5
        s = max(0, round(Int64, log(normA)/log(2), RoundToZero) + 2)
        A = A * 2.0^(-s) # scale!(A, 2.0^(-s))
    end

    # Horner evaluation of the irreducible fraction
    A2 = A * A
    Q = c[p+1]*Matrix{eltype(A)}(I, size(A))
    P = c[p]*Matrix{eltype(A)}(I, size(A))
    odd = 1
    @inbounds begin 
        for k = p-1:-1:1
            if odd == 1
                Q = Q * A2 + c[k] * I
            else
                P = P * A2 + c[k] * I
            end
            odd = 1 - odd
        end
    end

    if odd == 1
        Q = Q * A
        Q = Q - P
        E = -(I + 2 * \(Q, Matrix(P)))
    else
        P = P * A
        Q = Q - P
        E = I + 2 * \(Q, Matrix(P))
    end

    # squaring
    @inbounds begin 
        for k = 1:s
            E = E * E
        end
    end

    return E

end # padm
