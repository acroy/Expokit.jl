[![Build Status](https://travis-ci.org/acroy/Expokit.jl.png)](https://travis-ci.org/acroy/Expokit.jl)

# Expokit

This package provides Julia implementations of some routines contained
in [EXPOKIT](http://www.maths.uq.edu.au/expokit). Those routines allow
an efficient calculation of the action of matrix exponentials on vectors
for large sparse matrices. For more details about the methods see
*R.B. Sidje, ACM Trans. Math. Softw., 24(1):130-156, 1998*
(or [its preprint](http://www.expokit.org/paper.pdf)).

**Note:** Currently only `expmv` (which is called `expv` in EXPOKIT) is available.

## Usage
```julia
Pkg.clone("git://github.com/acroy/Expokit.jl.git")
```

## expmv

```julia
w = expmv!{T}( w::Vector{T}, t::Real, A, v::Vector{T}; kwargs...)
```
The function `expmv!` calculates `w = exp(t*A)*v`, where `A` is a
matrix or any type that supports `norm`, `size` and `A_mul_B!` and `v` a dense vector by using Krylov subspace projections. The result is
stored in `w`.

The following keywords are supported
- `tol`: tolerance to control step size
- `m`: size of Krylov subspace
- `norm`: user-supplied norm

For convenience, the following versions of `expmv` are provided
```julia
v = expmv!{T}( t::Real, A, v::Vector{T}; kwargs...)
w = expmv{T}( t::Real, A, v::Vector{T}; kwargs...)
```
