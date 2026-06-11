"""
    FFCMConfig{T}(
        ; L, R_c, N, a = T(1), kernel_widths_ratio, num_grid_points, M_G, viscosity
    ) -> FFCMConfig{T}

Configuration of the Fast FCM mobility operator: the cell partition of the pairwise
correction (paper §4), the FCM grid parameters (paper §3 and §5), and the per-call buffers
sized for `N` particles. Built once and reused across `mobility!` calls, which write into
the buffers this struct owns. `N` is fixed at construction — changing `N` requires a new
`FFCMConfig`.

# Keywords
- `L::NTuple{3, T}`: the domain lengths `(L_x, L_y, L_z)` (paper §4). Each component must be
  positive.
- `R_c::T`: the cutoff radius of the pairwise correction. Must satisfy
  `0 < R_c ≤ min(L)/2`, so the minimum image is unambiguous and no particle is corrected
  against its own periodic image (paper §4).
- `N::Integer`: the number of particles; fixes the hot-path buffer sizes. Must be positive.
- `a::T = T(1)`: the particle hydrodynamic radius. Only `a == T(1)` is currently supported;
  the kernel width follows `σ = a/√π` (paper §2, the FCM radius-width relation that recovers
  the single-particle Stokes drag).
- `kernel_widths_ratio::T`: the ratio between the wider Gaussian kernel width and the
  standard FCM kernel width (paper §5). Must satisfy `kernel_widths_ratio ≥ T(1)`; ratio of
  1 is the standard FCM limit.
- `num_grid_points::NTuple{3, Int32}`: the FFT grid dimensions `(M_x, M_y, M_z)` (paper §3).
  The induced spacing `h = L_i / M_i` must be identical across axes (paper §3 assumption).
- `M_G::Integer`: the cubic stencil support per axis (paper §5). Stored as `Int32`; must
  satisfy `2 ≤ M_G ≤ min(num_grid_points)` — the stencil cannot be wider than the grid on
  any axis, or the periodic wrap would alias distinct stencil points onto the same grid
  point.
- `viscosity::T`: the fluid dynamic viscosity (paper §2, Stokes momentum balance). Must be
  positive.

# Returns
- `FFCMConfig{T}`: the configuration `mobility!` operates on.

# Throws
- `ArgumentError`: if `R_c ≤ 0`, any `L_i ≤ 0`, `R_c > min(L)/2`, `N ≤ 0`, `a ≠ T(1)`
  (non-unit radius not yet implemented), `kernel_widths_ratio < 1`, `M_G < 2`,
  `M_G > min(num_grid_points)`, any `num_grid_points` component `< 1`, the grid spacing is
  anisotropic, or `viscosity ≤ 0`.

See `spec/spatial-hashing.md`, `spec/particle-sorting.md`, `spec/force-spreading.md`, and
`spec/stokes-solve.md`.
"""
struct FFCMConfig{T <: AbstractFloat, GridField, SpectralField, FwdTransform, InvTransform}
    L::NTuple{3, T}
    R_c::T
    num_cells::NTuple{3, Int32}
    cell_size::NTuple{3, T}
    inv_cell_size::NTuple{3, T}
    cell_hash::Vector{Int32}
    original_index::Vector{Int32}
    cell_start::Vector{Int32}
    cell_end::Vector{Int32}
    counting_sort_scratch::Vector{Int32}
    Y_sorted::Matrix{T}
    F_sorted::Matrix{T}
    Y_wrapped::Matrix{T}
    a::T
    σ::T
    Σ::T
    num_grid_points::NTuple{3, Int32}
    M_G::Int32
    h::T
    inv_h::T
    force_density::GridField
    gaussian_x::Vector{T}
    gaussian_y::Vector{T}
    gaussian_z::Vector{T}
    r²_x::Vector{T}
    r²_y::Vector{T}
    r²_z::Vector{T}
    idx_x::Vector{Int32}
    idx_y::Vector{Int32}
    idx_z::Vector{Int32}
    μ::T
    fluid_velocity::GridField
    fluid_hat::SpectralField
    k_x::Vector{T}
    k_y::Vector{T}
    k_z::Vector{T}
    forward_fourier_transform::FwdTransform
    inverse_fourier_transform::InvTransform
end

"""
    _wrap_wavevectors(M, L) -> Vector{T}

Returns the per-axis Fourier wavevectors `k = 2π m / L` for a full transform axis of length
`M`, in FFTW's wrap-around order: the signed mode index is `m = j - 1` on the leading
non-negative half (`j ≤ M/2 + 1`) and `m = j - 1 - M` on the trailing negative half. Used
for the `y` and `z` axes of the Stokes-solve grid; the `x` axis keeps only the non-negative
half under the transform and is built inline.

# Arguments
- `M::Int32`: the number of grid points along the axis.
- `L::T`: the box length along the axis.

# Returns
- `Vector{T}` of length `M`: the per-axis wavevector components.

See `spec/stokes-solve.md`.
"""
function _wrap_wavevectors(M::Int32, L::T) where {T}
    twoπ = T(2) * T(π)
    M_half_plus_one = M ÷ Int32(2) + Int32(1)
    return T[twoπ * (j ≤ M_half_plus_one ? T(j - 1) : T(j - 1 - M)) / L for j in 1:M]
end

function FFCMConfig{T}(;
    L::NTuple{3, T},
    R_c::T,
    N::Integer,
    a::T = T(1),
    kernel_widths_ratio::T,
    num_grid_points::NTuple{3, Int32},
    M_G::Integer,
    viscosity::T,
) where {T <: AbstractFloat}
    R_c > zero(T) || throw(ArgumentError("R_c must be positive"))
    all(>(zero(T)), L) || throw(ArgumentError("L components must be positive"))
    R_c ≤ minimum(L) / T(2) || throw(ArgumentError(
        "R_c must be at most half the smallest box length;" *
        "got R_c = $(R_c), min(L)/2 = $(minimum(L) / T(2))",
    ))
    N > 0 || throw(ArgumentError("N must be positive"))
    a == T(1) || throw(ArgumentError(
        "non-unit particle radius not yet implemented; only a = 1 is " *
        "supported, got a = $(a)",
    ))
    kernel_widths_ratio ≥ T(1) || throw(ArgumentError(
        "kernel widths ratio must be at least 1; got $(kernel_widths_ratio)",
    ))
    M_G ≥ 2 || throw(ArgumentError("M_G must be at least 2; got $(M_G)"))
    all(≥(Int32(1)), num_grid_points) ||
        throw(ArgumentError("num_grid_points components must each be ≥ 1"))
    M_G ≤ minimum(num_grid_points) || throw(ArgumentError(
        "M_G must not exceed the grid on any axis; got M_G = $(M_G), " *
        "min(num_grid_points) = $(minimum(num_grid_points))",
    ))
    viscosity > zero(T) || throw(ArgumentError(
        "viscosity must be positive; got $(viscosity)",
    ))

    num_cells = ntuple(i -> max(floor(Int32, L[i] / R_c), Int32(3)), 3)
    cell_size = ntuple(i -> L[i] / num_cells[i], 3)
    inv_cell_size = ntuple(i -> one(T) / cell_size[i], 3)
    cell_hash = Vector{Int32}(undef, N)
    num_cells_total = prod(Int, num_cells)
    original_index = Vector{Int32}(undef, N)
    cell_start = Vector{Int32}(undef, num_cells_total)
    cell_end = Vector{Int32}(undef, num_cells_total)
    counting_sort_scratch = Vector{Int32}(undef, num_cells_total)
    Y_sorted = Matrix{T}(undef, 3, N)
    F_sorted = Matrix{T}(undef, 3, N)
    # Scratch the assembled `mobility!` driver folds the caller's positions into,
    # so the caller's `Y` is never mutated (see `mobility!`, spec/mobility.md).
    Y_wrapped = Matrix{T}(undef, 3, N)

    σ = a / sqrt(T(π))
    Σ = kernel_widths_ratio * σ

    h_per_axis = ntuple(i -> L[i] / num_grid_points[i], 3)
    rel_tol = sqrt(eps(T))
    isotropic = abs(h_per_axis[2] - h_per_axis[1]) ≤ rel_tol * h_per_axis[1] &&
                abs(h_per_axis[3] - h_per_axis[1]) ≤ rel_tol * h_per_axis[1]
    isotropic || throw(ArgumentError(
        "anisotropic grid spacing not supported (paper §3 assumes uniform h); " *
        "L_i/M_i = $(h_per_axis)",
    ))
    h = h_per_axis[1]
    inv_h = one(T) / h

    M_x, M_y, M_z = num_grid_points
    fx = zeros(T, M_x, M_y, M_z)
    fy = zeros(T, M_x, M_y, M_z)
    fz = zeros(T, M_x, M_y, M_z)
    force_density = StructArray{SVector{3, T}}((fx, fy, fz))

    M_G_i32 = Int32(M_G)
    gaussian_x = Vector{T}(undef, M_G_i32)
    gaussian_y = Vector{T}(undef, M_G_i32)
    gaussian_z = Vector{T}(undef, M_G_i32)
    r²_x = Vector{T}(undef, M_G_i32)
    r²_y = Vector{T}(undef, M_G_i32)
    r²_z = Vector{T}(undef, M_G_i32)
    idx_x = Vector{Int32}(undef, M_G_i32)
    idx_y = Vector{Int32}(undef, M_G_i32)
    idx_z = Vector{Int32}(undef, M_G_i32)

    ux = zeros(T, M_x, M_y, M_z)
    uy = zeros(T, M_x, M_y, M_z)
    uz = zeros(T, M_x, M_y, M_z)
    fluid_velocity = StructArray{SVector{3, T}}((ux, uy, uz))

    fft_M_x = M_x ÷ Int32(2) + Int32(1)
    fh_x = zeros(Complex{T}, fft_M_x, M_y, M_z)
    fh_y = zeros(Complex{T}, fft_M_x, M_y, M_z)
    fh_z = zeros(Complex{T}, fft_M_x, M_y, M_z)
    fluid_hat = StructArray{SVector{3, Complex{T}}}((fh_x, fh_y, fh_z))

    # Fourier wavevectors in FFTW's layout. The transform keeps only the
    # non-negative x-frequencies (indices `1:fft_M_x = M_x÷2 + 1`); the full y/z
    # axes use the wrap-around order built by `_wrap_wavevectors`.
    k_x = T[T(2) * T(π) * (i - 1) / L[1] for i in 1:fft_M_x]
    k_y = _wrap_wavevectors(M_y, L[2])
    k_z = _wrap_wavevectors(M_z, L[3])

    forward_fourier_transform = plan_rfft(fx)
    inverse_fourier_transform = plan_brfft(fh_x, Int(M_x))

    return FFCMConfig{
        T,
        typeof(force_density),
        typeof(fluid_hat),
        typeof(forward_fourier_transform),
        typeof(inverse_fourier_transform),
    }(
        L,
        R_c,
        num_cells,
        cell_size,
        inv_cell_size,
        cell_hash,
        original_index,
        cell_start,
        cell_end,
        counting_sort_scratch,
        Y_sorted,
        F_sorted,
        Y_wrapped,
        a,
        σ,
        Σ,
        num_grid_points,
        M_G_i32,
        h,
        inv_h,
        force_density,
        gaussian_x,
        gaussian_y,
        gaussian_z,
        r²_x,
        r²_y,
        r²_z,
        idx_x,
        idx_y,
        idx_z,
        viscosity,
        fluid_velocity,
        fluid_hat,
        k_x,
        k_y,
        k_z,
        forward_fourier_transform,
        inverse_fourier_transform,
    )
end
