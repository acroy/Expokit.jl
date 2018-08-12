export chbv, chbv!

const α0 = 0.183216998528140087E-11

#=
Negative of the coefficients and poles of the partial fraction expansion of the
uniform rational Chebyshev approximation of type (14, 14) to the exponential
function over the negative real axis.
=#
const α = [-0.557503973136501826E+02 + im*0.204295038779771857E+03,
            0.938666838877006739E+02 - im*0.912874896775456363E+02,
           -0.469965415550370835E+02 + im*0.116167609985818103E+02,
            0.961424200626061065E+01 + im*0.264195613880262669E+01,
           -0.752722063978321642E+00 - im*0.670367365566377770E+00,
            0.188781253158648576E-01 + im*0.343696176445802414E-01,
           -0.143086431411801849E-03 - im*0.287221133228814096E-03]

const θ = [0.562314417475317895E+01 - im*0.119406921611247440E+01,
           0.508934679728216110E+01 - im*0.358882439228376881E+01,
           0.399337136365302569E+01 - im*0.600483209099604664E+01,
           0.226978543095856366E+01 - im*0.846173881758693369E+01,
          -0.208756929753827868E+00 - im*0.109912615662209418E+02,
          -0.370327340957595652E+01 - im*0.136563731924991884E+02,
          -0.889777151877331107E+01 - im*0.166309842834712071E+02]

const αconj = [-0.557503973136501826E+02 - im*0.204295038779771857E+03,
                0.938666838877006739E+02 + im*0.912874896775456363E+02,
               -0.469965415550370835E+02 - im*0.116167609985818103E+02,
                0.961424200626061065E+01 - im*0.264195613880262669E+01,
               -0.752722063978321642E+00 + im*0.670367365566377770E+00,
                0.188781253158648576E-01 - im*0.343696176445802414E-01,
               -0.143086431411801849E-03 + im*0.287221133228814096E-03]

const θconj = [0.562314417475317895E+01 + im*0.119406921611247440E+01,
               0.508934679728216110E+01 + im*0.358882439228376881E+01,
               0.399337136365302569E+01 + im*0.600483209099604664E+01,
               0.226978543095856366E+01 + im*0.846173881758693369E+01,
              -0.208756929753827868E+00 + im*0.109912615662209418E+02,
              -0.370327340957595652E+01 + im*0.136563731924991884E+02,
              -0.889777151877331107E+01 + im*0.166309842834712071E+02]

"""
    chbv(A, vec)

Calculate matrix exponential acting on some vector using the Chebyshev method.

# Input

- `A`   -- matrix which can be dense or sparse
- `vec` -- vector on which the matrix exponential of `A` is applied

# Notes

This Julia implementation is based on Expokit's CHBV Matlab code by
Roger B. Sidje, see below.

---

    y = chbv( H, x )
    CHBV computes the direct action of the matrix exponential on
    a vector: y = exp(H) * x. It uses the partial fraction expansion of
    the uniform rational Chebyshev approximation of type (14,14).
    About 14-digit accuracy is expected if the matrix H is symmetric
    negative definite. The algorithm may behave poorly otherwise.
    See also PADM, EXPOKIT.

    Roger B. Sidje (rbs@maths.uq.edu.au)
    EXPOKIT: Software Package for Computing Matrix Exponentials.
    ACM - Transactions On Mathematical Software, 24(1):130-156, 1998
"""
function chbv(A, vec::Vector{T}) where {T}
    result = convert(Vector{promote_type(eltype(A), T)}, copy(vec))
    return chbv!(result, A, vec)
end

chbv!(A, vec::Vector{T}) where {T} = chbv!(vec, A, copy(vec))

function chbv!(w::Vector{T}, A, vec::Vector{T}) where {T<:Real}
    p = min(length(θ), length(α))
    rmul!(copyto!(w, vec), α0)
    @inbounds for i = 1:p
        w .+= real((A - θ[i]*I) \ (α[i] * vec))
    end
    return w
end

function chbv!(w::Vector{T}, A, vec::Vector{T}) where {T<:Complex}
    p = min(length(θ), length(α))
    rmul!(copyto!(w, vec), α0)
    t = [θ; θconj]
    a = 0.5 * [α; αconj]
    @inbounds for i = 1:2*p
        w .+= (A - t[i]*I) \ (a[i] * vec)
    end
    return w
end # chbv!
