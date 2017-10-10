export padm

"""
Computes the matrix exponential exp(A) of a large sparse matrix  using
Pade approximants. 

This Julia implementation originated from Expokit's PADM Matlab code by
Roger B. Sidje, see the original description below, and its license.

AUTHORS: 

- Marcelo Forets (2017-07-10) : initial implementation

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
const PADE_COEFFS = convert(Vector{Float64}, [1//1,
                              1//2,
                              8188362958855447//72057594037927936,
                              8734253822779143//576460752303423488,
                              2911417940926381//2305843009213693952,
                              145570897046319//2305843009213692962,
                              2797136075159//1860878688081779609])

"""
    padm(A, [p])

Compute the matrix exponential of a sparse matrix using the irreducible
(p, p)-degree rational Pade approximation to the exponential function.

# Input

- `A` -- sparse matrix
- `p` -- (optional, default: 6) degree of the rational Pade approximation to
         the exponential function
"""
function padm(A::SparseMatrixCSC{Float64, Int64}, p::Int64=6)::SparseMatrixCSC{Float64, Int64}

n = size(A, 1)

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
normA = norm(A, Inf)
s = 0
if normA > 0.5
    s = max(0, round(Int64, log(normA)/log(2), RoundToZero) + 2)
    A = A * 2.0^(-s) # scale!(A, 2.0^(-s))
end

# Horner evaluation of the irreducible fraction
A2 = A * A
Q = sparse(c[p+1] * I, n, n)  # I is a UniformScaling
P = sparse(c[p] * I, n, n)
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
    E = -(I + 2 * \(Q, full(P)))
else
    P = P * A
    Q = Q - P
    E = I + 2 * \(Q, full(P))
end

# squaring
@inbounds begin 
    for k = 1:s
        E = E * E
    end
end

return E

end # padm
