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

The grid spacing `Δx = L_i / num_grid_points[i]` is required to be
identical across axes (paper §3 isotropy assumption); the constructor
throws `ArgumentError` otherwise.

See `spec/spatial-hashing.md`, `spec/particle-sorting.md`,
`spec/force-spreading.md`.
"""
struct FFCMConfig{T <: AbstractFloat, FG}
    L::NTuple{3, T}
    R_c::T
    num_cells::NTuple{3, Int32}
    cell_size::NTuple{3, T}
    inv_cell_size::NTuple{3, T}
    cell_hash::Vector{Int32}
    original_index::Vector{Int32}
    cell_start::Vector{Int32}
    cell_end::Vector{Int32}
    cell_cursor::Vector{Int32}
    Y_sorted::Matrix{T}
    F_sorted::Matrix{T}
    a::T
    σ::T
    Σ::T
    num_grid_points::NTuple{3, Int32}
    M_G::Int32
    Δx::T
    inv_Δx::T
    force_grid::FG
    gauss_x::Vector{T}
    gauss_y::Vector{T}
    gauss_z::Vector{T}
    r²_x::Vector{T}
    r²_y::Vector{T}
    r²_z::Vector{T}
    ind_x::Vector{Int32}
    ind_y::Vector{Int32}
    ind_z::Vector{Int32}
end

function FFCMConfig{T}(;
    L::NTuple{3, T},
    R_c::T,
    N::Integer,
    a::T = T(1),
    Σ_over_σ::T,
    num_grid_points::NTuple{3, Int32},
    M_G::Integer,
) where {T <: AbstractFloat}
    R_c > zero(T) || throw(ArgumentError("R_c must be positive"))
    all(>(zero(T)), L) || throw(ArgumentError("L components must be positive"))
    N > 0 || throw(ArgumentError("N must be positive"))
    a == T(1) || error("non-unit particle radius not yet implemented")
    Σ_over_σ ≥ T(1) || throw(ArgumentError(
        "Σ/σ must be at least 1 (paper eq 267 requires Σ ≥ σ; the equality " *
        "case is the standard-FCM degenerate limit); got Σ/σ = $(Σ_over_σ)",
    ))
    M_G ≥ 2 || throw(ArgumentError("M_G must be at least 2; got $(M_G)"))
    all(≥(Int32(1)), num_grid_points) ||
        throw(ArgumentError("num_grid_points components must each be ≥ 1"))

    num_cells = ntuple(i -> max(floor(Int32, L[i] / R_c), Int32(3)), 3)
    cell_size = ntuple(i -> L[i] / num_cells[i], 3)
    inv_cell_size = ntuple(i -> one(T) / cell_size[i], 3)
    cell_hash = Vector{Int32}(undef, N)
    num_cells_total = prod(Int, num_cells)
    original_index = Vector{Int32}(undef, N)
    cell_start = Vector{Int32}(undef, num_cells_total)
    cell_end = Vector{Int32}(undef, num_cells_total)
    cell_cursor = Vector{Int32}(undef, num_cells_total)
    Y_sorted = Matrix{T}(undef, 3, N)
    F_sorted = Matrix{T}(undef, 3, N)

    σ = a / sqrt(T(π))
    Σ = Σ_over_σ * σ

    Δx_per_axis = ntuple(i -> L[i] / num_grid_points[i], 3)
    rel_tol = sqrt(eps(T))
    isotropic = abs(Δx_per_axis[2] - Δx_per_axis[1]) ≤ rel_tol * Δx_per_axis[1] &&
                abs(Δx_per_axis[3] - Δx_per_axis[1]) ≤ rel_tol * Δx_per_axis[1]
    isotropic || throw(ArgumentError(
        "anisotropic grid spacing not supported (paper §3 assumes uniform Δx); " *
        "L_i/M_i = $(Δx_per_axis)",
    ))
    Δx = Δx_per_axis[1]
    inv_Δx = one(T) / Δx

    M_x, M_y, M_z = num_grid_points
    fx = zeros(T, M_x, M_y, M_z)
    fy = zeros(T, M_x, M_y, M_z)
    fz = zeros(T, M_x, M_y, M_z)
    force_grid = StructArray{SVector{3, T}}((fx, fy, fz))

    M_G_i32 = Int32(M_G)
    gauss_x = Vector{T}(undef, M_G_i32)
    gauss_y = Vector{T}(undef, M_G_i32)
    gauss_z = Vector{T}(undef, M_G_i32)
    r²_x = Vector{T}(undef, M_G_i32)
    r²_y = Vector{T}(undef, M_G_i32)
    r²_z = Vector{T}(undef, M_G_i32)
    ind_x = Vector{Int32}(undef, M_G_i32)
    ind_y = Vector{Int32}(undef, M_G_i32)
    ind_z = Vector{Int32}(undef, M_G_i32)

    return FFCMConfig{T, typeof(force_grid)}(
        L,
        R_c,
        num_cells,
        cell_size,
        inv_cell_size,
        cell_hash,
        original_index,
        cell_start,
        cell_end,
        cell_cursor,
        Y_sorted,
        F_sorted,
        a,
        σ,
        Σ,
        num_grid_points,
        M_G_i32,
        Δx,
        inv_Δx,
        force_grid,
        gauss_x,
        gauss_y,
        gauss_z,
        r²_x,
        r²_y,
        r²_z,
        ind_x,
        ind_y,
        ind_z,
    )
end
