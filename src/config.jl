"""
    FFCMConfig{T}(; L, R_c, N, a = T(1), Σ_over_σ, num_grid_points, M_G)

Cold-path configuration of the Fast FCM mobility operator. Owns the cell
geometry derived from the periodic domain `L = (L_x, L_y, L_z)` (paper §4)
and the cutoff `R_c` for the pairwise correction, plus the FCM grid
parameters of paper §3 (`outline.tex:175`) and §5 (`outline.tex:571-573`)
and the hot-path buffers sized for `N` particles. The configuration is
built once and reused across many `mobility!` calls; all per-call work
writes into pre-allocated buffers owned by this struct. `N` is fixed at
construction — changing `N` requires a new `FFCMConfig`.

Keyword arguments for step 3 (force spreading):

- `a::T = T(1)` — particle hydrodynamic radius. Only `a == T(1)` is
  currently supported (paper §2, eq 163; CLAUDE.md unit-radius
  convention).
- `Σ_over_σ::T` — kernel resolution ratio Σ/σ (paper §5,
  `outline.tex:573`); must satisfy `Σ_over_σ ≥ T(1)`. The equality case
  `Σ = σ` is the standard-FCM degenerate limit.
- `num_grid_points::NTuple{3, Int32}` — FFT grid dimensions
  `(M_x, M_y, M_z)` (paper §3, `outline.tex:175`).
- `M_G::Integer` — cubic stencil support per axis (paper §5,
  `outline.tex:571`). Stored as `Int32`. Must be at least 2.

The grid spacing `h = L_i / num_grid_points[i]` is required to be
identical across axes (paper §3 isotropy assumption); the constructor
throws `ArgumentError` otherwise.

The cutoff must satisfy `0 < R_c ≤ min(L) / 2` so the pairwise correction's
minimum image is unambiguous and no particle is corrected against its own
periodic image (paper §4, `outline.tex:318`).

See `spec/spatial-hashing.md`, `spec/particle-sorting.md`,
`spec/force-spreading.md`.
"""
struct FFCMConfig{T <: AbstractFloat, FG, FH, FwdTransform, BwdTransform}
    L::NTuple{3, T}
    R_c::T
    num_cells::NTuple{3, Int32}
    cell_size::NTuple{3, T}
    inv_cell_size::NTuple{3, T}
    cell_hash::Vector{Int32}
    original_index::Vector{Int32}
    cell_start::Vector{Int32}
    cell_end::Vector{Int32}
    next_free_slot::Vector{Int32}
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
    force_density::FG
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
    fluid_velocity::FG
    fluid_hat::FH
    k_x::Vector{T}
    k_y::Vector{T}
    k_z::Vector{T}
    forward_fourier_transform::FwdTransform
    inverse_fourier_transform::BwdTransform
end

function FFCMConfig{T}(;
    L::NTuple{3, T},
    R_c::T,
    N::Integer,
    a::T = T(1),
    Σ_over_σ::T,
    num_grid_points::NTuple{3, Int32},
    M_G::Integer,
    μ::T,
) where {T <: AbstractFloat}
    R_c > zero(T) || throw(ArgumentError("R_c must be positive"))
    all(>(zero(T)), L) || throw(ArgumentError("L components must be positive"))
    R_c ≤ minimum(L) / T(2) || throw(ArgumentError(
        "R_c must be at most half the smallest box length (paper outline.tex:318 " *
        "requires the minimum image to be unambiguous, with no self-image " *
        "corrections); got R_c = $(R_c), min(L)/2 = $(minimum(L) / T(2))",
    ))
    N > 0 || throw(ArgumentError("N must be positive"))
    a == T(1) || error("non-unit particle radius not yet implemented")
    Σ_over_σ ≥ T(1) || throw(ArgumentError(
        "Σ/σ must be at least 1 (paper eq 267 requires Σ ≥ σ; the equality " *
        "case is the standard-FCM degenerate limit); got Σ/σ = $(Σ_over_σ)",
    ))
    M_G ≥ 2 || throw(ArgumentError("M_G must be at least 2; got $(M_G)"))
    all(≥(Int32(1)), num_grid_points) ||
        throw(ArgumentError("num_grid_points components must each be ≥ 1"))
    μ > zero(T) || throw(ArgumentError(
        "μ must be positive (paper §2 eq 152 requires a positive viscosity); " *
        "got μ = $(μ)",
    ))

    num_cells = ntuple(i -> max(floor(Int32, L[i] / R_c), Int32(3)), 3)
    cell_size = ntuple(i -> L[i] / num_cells[i], 3)
    inv_cell_size = ntuple(i -> one(T) / cell_size[i], 3)
    cell_hash = Vector{Int32}(undef, N)
    num_cells_total = prod(Int, num_cells)
    original_index = Vector{Int32}(undef, N)
    cell_start = Vector{Int32}(undef, num_cells_total)
    cell_end = Vector{Int32}(undef, num_cells_total)
    next_free_slot = Vector{Int32}(undef, num_cells_total)
    Y_sorted = Matrix{T}(undef, 3, N)
    F_sorted = Matrix{T}(undef, 3, N)
    # Scratch the assembled `mobility!` driver folds the caller's positions into,
    # so the caller's `Y` is never mutated (see `mobility!`, spec/mobility.md).
    Y_wrapped = Matrix{T}(undef, 3, N)

    σ = a / sqrt(T(π))
    Σ = Σ_over_σ * σ

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

    PI2 = T(2) * T(π)
    k_x = T[PI2 * (i - 1) / L[1] for i in 1:fft_M_x]
    k_y = T[
        PI2 * (j ≤ M_y ÷ Int32(2) + Int32(1) ? T(j - 1) : T(j - 1 - M_y)) / L[2]
        for j in 1:M_y
    ]
    k_z = T[
        PI2 * (k ≤ M_z ÷ Int32(2) + Int32(1) ? T(k - 1) : T(k - 1 - M_z)) / L[3]
        for k in 1:M_z
    ]

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
        next_free_slot,
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
        μ,
        fluid_velocity,
        fluid_hat,
        k_x,
        k_y,
        k_z,
        forward_fourier_transform,
        inverse_fourier_transform,
    )
end
