module FFCMCUDAExt

# CUDA backend of the Fast FCM mobility operator (Su & Keaveny 2024, §3 and §4).
# Loaded automatically when a user runs `using CUDA` alongside `using FFCM`,
# through the package's `[weakdeps]`/`[extensions]` tables. This is the only place
# CUDA appears; `src` stays free of it (CLAUDE.md §7). See spec/cuda-conventions.md.
#
# The extension mirrors the `src/` layout, one file per pipeline concern:
# `config.jl` builds the device buffers and cuFFT plans; `cell_list.jl` holds the
# step-1 device kernels (wrap, hash, counting sort, gather).

using FFCM
using CUDA
using StaticArrays: SVector
using StructArrays: StructArray, components

import FFCM:
    _assemble_gpu_buffers,
    _assign_cells_kernel!,
    _build_cell_list_kernel!,
    _gather_particles_kernel!,
    _modified_kernel_coefficients,
    _spread_forces_kernel!,
    wrap_positions!

include("config.jl")
include("cell_list.jl")
include("spread_forces.jl")

end # module FFCMCUDAExt
