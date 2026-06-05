"""
    wrap_positions!(dest, src, L) -> dest
    wrap_positions!(Y, L) -> Y

Fold each column of the `3×N` position matrix `src` into the canonical periodic
domain `[0, L_i)` for each axis `i`, writing the result into `dest`. The
out-of-place form lets the assembled `mobility!` driver wrap into a scratch
buffer without touching the caller's positions; the in-place form
(`dest === src`) folds `Y` in place. Idempotent.

# Arguments
- `dest::AbstractMatrix{T}`: the `3×N` destination matrix (may alias `src`).
- `src::AbstractMatrix{T}`: the `3×N` source positions.
- `Y::AbstractMatrix{T}`: in the two-argument form, the matrix folded in place.
- `L::NTuple{3, T}`: the periodic box lengths.

# Returns
- `dest` (or `Y`): the same matrix, with every column in `[0, L_i)`.

# Notes
The fp-roundoff corner case where `mod(y, L_i)` rounds to exactly `L_i` is left
to the downstream cell-index clamp in `_assign_cells_kernel!`.

See `spec/spatial-hashing.md`.
"""
function wrap_positions!(
    dest::AbstractMatrix{T}, src::AbstractMatrix{T}, L::NTuple{3, T},
) where {T}
    @inbounds @simd for n in axes(src, 2)
        dest[1, n] = mod(src[1, n], L[1])
        dest[2, n] = mod(src[2, n], L[2])
        dest[3, n] = mod(src[3, n], L[3])
    end
    return dest
end

wrap_positions!(Y::AbstractMatrix{T}, L::NTuple{3, T}) where {T} =
    wrap_positions!(Y, Y, L)

"""
    _build_cell_list_kernel!(
        original_index, cell_start, cell_end, next_free_slot, cell_hash,
    ) -> original_index

Function-barrier kernel that counting-sorts the `N` particles by their 0-based
`cell_hash` into `original_index` (sorted slot `s` → original particle index
`n`, read as `n = original_index[s]`), and fills the per-cell 1-based inclusive
index ranges `cell_start[c] : cell_end[c]` into the sorted order, for cells
`c ∈ 0:total-1` stored at array index `c + 1`.

The sort is stable: particles sharing a cell keep their original relative order.
An empty cell `c` yields `cell_end[c] = cell_start[c] - 1`, i.e. an empty range.

# Arguments
- `original_index::Vector{Int32}`: length-`N` output; sorted slot → original
  particle index.
- `cell_start::Vector{Int32}`, `cell_end::Vector{Int32}`: length-`total` output;
  the 1-based inclusive slot range per cell.
- `next_free_slot::Vector{Int32}`: length-`total` scratch for the counting sort.
- `cell_hash::Vector{Int32}`: length-`N` input; each particle's 0-based cell.

# Returns
- `original_index`: the same vector, holding the sorted permutation.

# Notes
Preconditions (the caller guarantees these, so the loops are `@inbounds`):
`cell_hash[n] ∈ 0:total-1` for every particle `n`, where
`total = length(cell_start)`; `original_index` has length `N`; `cell_start`,
`cell_end`, `next_free_slot` all have length `total`.

See `spec/particle-sorting.md`.
"""
function _build_cell_list_kernel!(
    original_index::Vector{Int32},
    cell_start::Vector{Int32},
    cell_end::Vector{Int32},
    next_free_slot::Vector{Int32},
    cell_hash::Vector{Int32},
)
    total = length(cell_start)
    fill!(next_free_slot, Int32(0))
    @inbounds for n in eachindex(cell_hash)
        next_free_slot[cell_hash[n] + Int32(1)] += Int32(1)
    end
    acc = Int32(1)
    @inbounds for c in 1:total
        cell_start[c] = acc
        acc += next_free_slot[c]
        cell_end[c] = acc - Int32(1)
        next_free_slot[c] = cell_start[c]
    end
    @inbounds for n in eachindex(cell_hash)
        c = cell_hash[n] + Int32(1)
        original_index[next_free_slot[c]] = Int32(n)
        next_free_slot[c] += Int32(1)
    end
    return original_index
end

"""
    sort_particles_by_cell!(config, Y, F) -> config

Step 2 of the Fast FCM algorithm (Su & Keaveny 2024, §4). Counting-sorts the
particles by their cell hash and gathers their positions and forces into that
sorted order, so that particles sharing a cell are contiguous in memory.
Allocation-free and type-stable on `T`.

On return, `config` holds:

- `config.original_index` — sorted slot `s` → original particle index.
- `config.cell_start[c+1] : config.cell_end[c+1]` — 1-based inclusive range of
  sorted slots occupied by cell `c` (empty range if the cell is empty).
- `config.Y_sorted`, `config.F_sorted` — `Y` and `F` in sorted order.

# Arguments
- `config::FFCMConfig{T}`: the compiled configuration; the sort outputs are
  written into its buffers. Assumes `config.cell_hash` is current, i.e.
  `assign_cells!(config, Y)` ran since `Y` last changed.
- `Y::AbstractMatrix{T}`, `F::AbstractMatrix{T}`: the `3×N` positions and forces
  to gather into sorted order.

# Returns
- `config`: the same configuration, with the sort outputs populated.

See `spec/particle-sorting.md`.
"""
function sort_particles_by_cell!(
    config::FFCMConfig{T}, Y::AbstractMatrix{T}, F::AbstractMatrix{T},
) where {T}
    _build_cell_list_kernel!(
        config.original_index,
        config.cell_start,
        config.cell_end,
        config.next_free_slot,
        config.cell_hash,
    )
    _gather_particles_kernel!(
        config.Y_sorted, config.F_sorted, Y, F, config.original_index,
    )
    return config
end

"""
    _gather_particles_kernel!(Y_sorted, F_sorted, Y, F, original_index)
        -> Y_sorted

Function-barrier kernel that gathers the `3×N` positions `Y` and forces `F`
into sorted order under the permutation `original_index`, writing
`Y_sorted[:, s] = Y[:, original_index[s]]` and likewise for `F`. This puts
particles sharing a cell into contiguous columns, the memory locality the
paper's step 2 exists to provide.

# Arguments
- `Y_sorted::AbstractMatrix{T}`, `F_sorted::AbstractMatrix{T}`: the `3×N`
  outputs, written in sorted order.
- `Y::AbstractMatrix{T}`, `F::AbstractMatrix{T}`: the `3×N` source positions and
  forces in original order.
- `original_index::Vector{Int32}`: the sorted-slot → original-particle
  permutation.

# Returns
- `Y_sorted`: the same matrix, holding the gathered positions.

# Notes
Preconditions (caller-guaranteed, so the loop is `@inbounds`): all five arrays
have second dimension `N = length(original_index)`; `Y`, `F`, `Y_sorted`,
`F_sorted` have first dimension 3; `original_index[s] ∈ 1:N`.

See `spec/particle-sorting.md`.
"""
function _gather_particles_kernel!(
    Y_sorted::AbstractMatrix{T},
    F_sorted::AbstractMatrix{T},
    Y::AbstractMatrix{T},
    F::AbstractMatrix{T},
    original_index::Vector{Int32},
) where {T}
    @inbounds for s in eachindex(original_index)
        n = original_index[s]
        Y_sorted[1, s] = Y[1, n]
        Y_sorted[2, s] = Y[2, n]
        Y_sorted[3, s] = Y[3, n]
        F_sorted[1, s] = F[1, n]
        F_sorted[2, s] = F[2, n]
        F_sorted[3, s] = F[3, n]
    end
    return Y_sorted
end

"""
    assign_cells!(config, Y) -> config.cell_hash

Step 1 of the Fast FCM algorithm (Su & Keaveny 2024, §4). Write each particle's
cell index into `config.cell_hash`. The hash linearises the 3-D cell coordinate
with `x` fastest and `z` slowest.

# Arguments
- `config::FFCMConfig{T}`: the compiled configuration; `config.cell_hash` is
  overwritten.
- `Y::AbstractMatrix{T}`: the `3×N` positions, already folded into the canonical
  domain by `wrap_positions!`.

# Returns
- `config.cell_hash`: the per-particle 0-based cell indices.

See `spec/spatial-hashing.md`.
"""
function assign_cells!(config::FFCMConfig{T}, Y::AbstractMatrix{T}) where {T}
    _assign_cells_kernel!(
        config.cell_hash, Y, config.inv_cell_size, config.num_cells,
    )
    return config.cell_hash
end

"""
    _assign_cells_kernel!(cell_hash, Y, inv_cell_size, num_cells) -> cell_hash

Function-barrier kernel for `assign_cells!`. Writes
`cell_hash[n] = x_c + (y_c + z_c · m_y) · m_x` for each particle column of `Y`,
with cell coordinates `(x_c, y_c, z_c) = floor.(Y_n .* inv_cell_size)` and
`num_cells[i] = m_i`.

# Arguments
- `cell_hash::Vector{Int32}`: length-`N` output; each particle's 0-based cell.
- `Y::AbstractMatrix{T}`: the `3×N` positions, folded into `[0, L_i)`.
- `inv_cell_size::NTuple{3, T}`: the reciprocal cell sizes `1/Δ_i`.
- `num_cells::NTuple{3, Int32}`: the cell-grid dimensions `(m_x, m_y, m_z)`.

# Returns
- `cell_hash`: the same vector, holding the per-particle cell indices.

# Notes
`Y` is assumed already folded into `[0, L_i)` by `wrap_positions!`. The per-axis
`min(·, m_i - 1)` clamp defends against the fp-roundoff corner where
`Y_i · inv_cell_size_i` lands exactly at `m_i` and the floored cell index would
otherwise be one past the last valid cell.

See `spec/spatial-hashing.md`.
"""
function _assign_cells_kernel!(
    cell_hash::Vector{Int32},
    Y::AbstractMatrix{T},
    inv_cell_size::NTuple{3, T},
    num_cells::NTuple{3, Int32},
) where {T <: AbstractFloat}
    @inbounds @simd for n in axes(Y, 2)
        xc = floor(Int32, Y[1, n] * inv_cell_size[1])
        yc = floor(Int32, Y[2, n] * inv_cell_size[2])
        zc = floor(Int32, Y[3, n] * inv_cell_size[3])
        xc = min(xc, num_cells[1] - Int32(1))
        yc = min(yc, num_cells[2] - Int32(1))
        zc = min(zc, num_cells[3] - Int32(1))
        cell_hash[n] = xc + (yc + zc * num_cells[2]) * num_cells[1]
    end
    return cell_hash
end
