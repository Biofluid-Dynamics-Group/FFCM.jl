module FFCM

using StaticArrays: SVector
using StructArrays: StructArray, components
using FFTW: plan_rfft, plan_brfft
import LinearAlgebra: mul!, dot, issymmetric, isposdef
using SpecialFunctions: erf

include("config.jl")
include("cell_list.jl")
include("stencil.jl")
include("spread_forces.jl")
include("stokes_solve.jl")
include("interpolate.jl")
include("correct_velocities.jl")
include("mobility.jl")

export FFCMConfig, mobility!, FFCMMobility

end