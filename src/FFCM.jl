module FFCM

using StaticArrays: SVector
using StructArrays: StructArray, components
using FFTW: plan_rfft, plan_brfft
using LinearAlgebra: mul!
using SpecialFunctions: erf

include("config.jl")
include("cell_list.jl")
include("stencil.jl")
include("spread_forces.jl")
include("stokes_solve.jl")
include("interpolate.jl")
include("correct_velocities.jl")

export FFCMConfig, spread_forces!, stokes_solve!, interpolate_velocities!,
    correct_velocities!

end
