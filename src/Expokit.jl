module Expokit

using Compat
import Compat.view

const axpy! = Base.LinAlg.axpy!
const expm! = Base.LinAlg.expm!

include("expmv.jl")
include("padm.jl")
include("chbv.jl")
include("phimv.jl")

end # module
