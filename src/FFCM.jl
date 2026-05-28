module FFCM

using StaticArrays: SVector
using StructArrays: StructArray, components

include("config.jl")
include("cell_list.jl")
include("spread_forces.jl")

export FFCMConfig, spread_forces!

end
