# Expokit

This package provides Julia implementations of some routines contained
in [EXPOKIT](http://www.maths.uq.edu.au/expokit). Those routines allow
an efficient calculation of the action of matrix exponentials on vectors
for large sparse matrices. For more details about the methods see
*R.B. Sidje, ACM Trans. Math. Softw., 24(1):130-156, 1998*
(or [its preprint](http://www.expokit.org/paper.pdf)).

**Note:** Currently only `expv` is available.

## Usage

## expv

```julia
w = expv{T}( v::Vector{T}, t::Real, amat::AbstractMatrix; kwargs...)
```
The function `expv` calculates `w = exp(t*amat)*v`, where `amat` is a
matrix and `v` a vector by using Krylov subspace projections.

The following keywords are supported
- `tol`: tolerance to control step size
- `m`: size of Krylov subspace
- `norm`: user-supplied norm
