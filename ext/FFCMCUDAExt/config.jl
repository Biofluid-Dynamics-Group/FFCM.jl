"""
    _assemble_gpu_buffers(T, N, num_grid_points, M_G, num_cells_total, k_x, k_y, k_z,
        neighbor_map_host) -> (cells, particles, grid, solver, stencil)

Allocates the GPU-backed sub-struct buffers and the cuFFT plans, the device counterpart of
`FFCM._assemble_cpu_buffers`. Adding this method to the `_assemble_gpu_buffers` function
declared in `FFCM` is what makes `FFCMConfig(; …, gpu_acceleration = true)` build a
GPU-backed configuration; without this extension loaded, only the throwing fallback exists.

The host-built `neighbor_map_host` (the half-shell neighbour map, geometry-only) is copied to
the device and stored on `cells`; the CPU backend stores `nothing` there instead.

# Returns
- `Tuple` of `FFCM.CellBuffers`, `FFCM.ParticleBuffers`, `FFCM.GridBuffers`,
  `FFCM.SolverState`, `FFCM.StencilBuffers`, all backed by `CuArray` and cuFFT plans.

See `spec/cuda-conventions.md`.
"""
function _assemble_gpu_buffers(
    ::Type{T},
    N::Integer,
    num_grid_points::NTuple{3, Int32},
    M_G::Int32,
    num_cells_total::Integer,
    k_x::Vector{T},
    k_y::Vector{T},
    k_z::Vector{T},
    neighbor_map_host::Vector{Int32},
) where {T}
    cells = FFCM.CellBuffers(
        CuVector{Int32}(undef, N),
        CuVector{Int32}(undef, N),
        CuVector{Int32}(undef, num_cells_total),
        CuVector{Int32}(undef, num_cells_total),
        CuVector{Int32}(undef, num_cells_total),
        CuArray(neighbor_map_host),
    )
    particles = FFCM.ParticleBuffers(
        CuMatrix{T}(undef, 3, N),
        CuMatrix{T}(undef, 3, N),
        CuMatrix{T}(undef, 3, N),
        # Host↔device staging: device-resident copies of the caller's raw positions and
        # forces and of the velocity output, so `mobility!` accepts host arrays and hides the
        # traffic (spec/cuda-conventions.md, "Assembled operator: the host↔device boundary").
        CuMatrix{T}(undef, 3, N),
        CuMatrix{T}(undef, 3, N),
        CuMatrix{T}(undef, 3, N),
    )

    M_x, M_y, M_z = num_grid_points
    fx = CuArray{T, 3}(undef, M_x, M_y, M_z)
    fy = CuArray{T, 3}(undef, M_x, M_y, M_z)
    fz = CuArray{T, 3}(undef, M_x, M_y, M_z)
    ux = CuArray{T, 3}(undef, M_x, M_y, M_z)
    uy = CuArray{T, 3}(undef, M_x, M_y, M_z)
    uz = CuArray{T, 3}(undef, M_x, M_y, M_z)
    grid = FFCM.GridBuffers(
        StructArray{SVector{3, T}}((fx, fy, fz)),
        StructArray{SVector{3, T}}((ux, uy, uz)),
    )

    fft_M_x = M_x ÷ Int32(2) + Int32(1)
    fh_x = CuArray{Complex{T}, 3}(undef, fft_M_x, M_y, M_z)
    fh_y = CuArray{Complex{T}, 3}(undef, fft_M_x, M_y, M_z)
    fh_z = CuArray{Complex{T}, 3}(undef, fft_M_x, M_y, M_z)
    fluid_hat = StructArray{SVector{3, Complex{T}}}((fh_x, fh_y, fh_z))

    # cuFFT plans dispatch through the same `plan_rfft`/`plan_brfft` generics FFTW
    # extends. Unlike FFTW's measuring planner, cuFFT planning does not execute on
    # the buffers, so the buffers need no post-plan zeroing.
    forward_fourier_transform = FFCM.plan_rfft(fx)
    inverse_fourier_transform = FFCM.plan_brfft(fh_x, Int(M_x))
    solver = FFCM.SolverState(
        fluid_hat,
        CuArray(k_x),
        CuArray(k_y),
        CuArray(k_z),
        forward_fourier_transform,
        inverse_fourier_transform,
    )

    stencil = FFCM.StencilBuffers(
        StructArray{SVector{3, T}}((
            CuVector{T}(undef, M_G), CuVector{T}(undef, M_G), CuVector{T}(undef, M_G),
        )),
        StructArray{SVector{3, T}}((
            CuVector{T}(undef, M_G), CuVector{T}(undef, M_G), CuVector{T}(undef, M_G),
        )),
        StructArray{SVector{3, Int32}}((
            CuVector{Int32}(undef, M_G),
            CuVector{Int32}(undef, M_G),
            CuVector{Int32}(undef, M_G),
        )),
    )

    # The buffers are left unzeroed: every one is written before it is read on a
    # `mobility!` call (`force_density` is zeroed at the start of `spread_forces!`;
    # `fluid_velocity` and `fluid_hat` are overwritten by the transforms). Keeping
    # construction free of `fill!` keeps it free of any Julia GPU-kernel launch —
    # only device allocations and cuFFT plan creation, neither of which JIT-compiles
    # a kernel.
    return (cells, particles, grid, solver, stencil)
end
