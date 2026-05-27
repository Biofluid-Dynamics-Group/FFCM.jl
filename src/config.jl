"""
    FFCMConfig{T}(; L, R_c, N)

Cold-path configuration of the Fast FCM mobility operator. Owns the cell
geometry derived from the periodic domain `L = (L_x, L_y, L_z)` (paper §4)
and the cutoff `R_c`, plus the hot-path buffers sized for `N` particles.
The configuration is built once and reused across many `mobility!` calls;
all per-call work writes into pre-allocated buffers owned by this struct.
`N` is fixed at construction — changing `N` requires a new `FFCMConfig`.

See `spec/spatial-hashing.md`.
"""
struct FFCMConfig{T <: AbstractFloat}
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
end

function FFCMConfig{T}(;
    L::NTuple{3, T},
    R_c::T,
    N::Integer,
) where {T <: AbstractFloat}
    R_c > zero(T) || throw(ArgumentError("R_c must be positive"))
    all(>(zero(T)), L) || throw(ArgumentError("L components must be positive"))
    N > 0 || throw(ArgumentError("N must be positive"))
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
    return FFCMConfig{T}(
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
    )
end
