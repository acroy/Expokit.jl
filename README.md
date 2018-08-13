[![Build Status](https://travis-ci.org/acroy/Expokit.jl.png)](https://travis-ci.org/acroy/Expokit.jl)

# Expokit

This package provides Julia implementations of some routines contained
in [EXPOKIT](http://www.maths.uq.edu.au/expokit). Those routines allow
an efficient calculation of the action of matrix exponentials on vectors
for large sparse matrices. For more details about the methods see
*R.B. Sidje, ACM Trans. Math. Softw., 24(1):130-156, 1998*
(or [its preprint](http://www.maths.uq.edu.au/expokit/paper.pdf)).

**Note:** Apart from `expmv` (which is called `expv` in EXPOKIT) also `phimv`, `padm` and `chbv` are available.

## Usage
```julia
Pkg.add("Expokit")
```

## expmv

```julia
w = expmv!{T}( w::Vector{T}, t::Number, A, v::Vector{T}; kwargs...)
```
The function `expmv!` calculates `w = exp(t*A)*v`, where `A` is a
matrix or any type that supports `size`, `eltype` and `mul!` and `v` is a dense vector by using Krylov subspace projections. The result is
stored in `w`.

The following keywords are supported
- `tol`: tolerance to control step size (default: `1e-7`)
- `m`: size of Krylov subspace (default: `min(30,size(A,1))`)
- `norm`: user-supplied function to calculate vector norm (dafault: `Base.norm`)
- `anorm`: operator/matrix norm of `A` to estimate first time-step (default: `opnorm(A, Inf)`)

For convenience, the following versions of `expmv` are provided
```julia
v = expmv!{T}( t::Number, A, v::Vector{T}; kwargs...)
w = expmv{T}( t::Number, A, v::Vector{T}; kwargs...)
```

## phimv

```julia
w = phimv!{T}( w::Vector{T}, t::Number, A, u::Vector{T}, v::Vector{T}; kwargs...)
```
The function `phimv!` calculates `w = e^{tA}v + t φ(t A) u` with `φ(z) = (exp(z)-1)/z`, where `A` is a
matrix or any type that supports `size`, `eltype` and `mul!`, `u` and `v` are dense vectors by using Krylov subspace projections. The result is stored in `w`. The supported keywords are the same as for `expmv!`.

## chbv

```julia
chbv!{T}(w::Vector{T}, A, v::Vector{T})
```
The function `chbv!` calculates `w = exp(A)*v` using the partial fraction expansion of
the uniform rational Chebyshev approximation of type (14,14). 

## padm

```julia
padm(A; p=6)
```
The function `padm` calculates the matrix exponential `exp(A)` of `A` using the irreducible 
(p,p)-degree rational Pade approximation to the exponential function.

