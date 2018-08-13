__precompile__(true)

"""
Main module of `Expokit` -- Julia implementations of some routines contained
in [EXPOKIT](https://www.maths.uq.edu.au/expokit/).

Those routines allow an efficient calculation of the action of matrix exponentials
on vectors for large sparse matrices. For more details about the methods see
R.B. Sidje, ACM Trans. Math. Softw., 24(1):130-156, 1998 (or its
[preprint](https://www.maths.uq.edu.au/expokit/paper.pdf)).
"""
module Expokit

using LinearAlgebra

const axpy! = LinearAlgebra.axpy!
const exp! = LinearAlgebra.exp!

include("expmv.jl")
include("padm.jl")
include("chbv.jl")
include("phimv.jl")

end # module
