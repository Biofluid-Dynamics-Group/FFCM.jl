"""
    wrap_positions!(destination, source, L) -> destination
    wrap_positions!(Y, L) -> Y

Fold each column of the `3xN` position matrix `source` into the periodic domain `[0, L_i)`
for each axis `i`, writing the result into `destination`. A two-argument form  of the same
function folds `Y` in place.

# Arguments
- `destination::AbstractMatrix{T}`: the `3xN` destination matrix (may alias `source`).
- `source::AbstractMatrix{T}`: the `3xN` source positions.
- `Y::AbstractMatrix{T}`: in the two-argument form, the matrix folded in place.
- `L::NTuple{3, T}`: the periodic box lengths.

# Returns
- `destination` (or `Y`): the same matrix, with every column in `[0, L_i)`.

# Notes
The floating point roundoff corner case where `mod(y, L_i)` rounds to exactly `L_i` is left
to the downstream cell-index clamp in `_assign_cells_kernel!`.

See `spec/spatial-hashing.md`.
"""
function wrap_positions!(
    destination::AbstractMatrix{T}, source::AbstractMatrix{T}, L::NTuple{3, T},
) where {T}
    @inbounds @simd for n in axes(source, 2)
        destination[1, n] = mod(source[1, n], L[1])
        destination[2, n] = mod(source[2, n], L[2])
        destination[3, n] = mod(source[3, n], L[3])
    end
    return destination
end

wrap_positions!(Y::AbstractMatrix{T}, L::NTuple{3, T}) where {T} =
    wrap_positions!(Y, Y, L)

"""
    _build_cell_list_kernel!(
        original_index, cell_start, cell_end, counting_sort_scratch, cell_hash,
    ) -> original_index

Counting-sort the `N` particles by their 0-based `cell_hash` into `original_index` (sorted
slot `s` to original particle index `n`, read as `n = original_index[s]`), and fill the
per-cell 1-based inclusive index ranges `cell_start[c] : cell_end[c]` into the sorted order,
for cells `c ∈ 0:total-1` stored at array index `c + 1`.

Particles sharing a cell keep their original relative order. An empty cell `c` yields
`cell_end[c] = cell_start[c] - 1`, i.e. an empty range.

# Arguments
- `original_index::Vector{Int32}`: length-`N` output; sorted slot to original particle
  index.
- `cell_start::Vector{Int32}`, `cell_end::Vector{Int32}`: length-`total` output; the 1-based
  inclusive slot range per cell.
- `counting_sort_scratch::Vector{Int32}`: length-`total` scratch — first the per-cell
  particle count, then the per-cell write cursor.
- `cell_hash::Vector{Int32}`: length-`N` input; each particle's 0-based cell.

# Returns
- `original_index`: the same vector, holding the sorted permutation.

# Notes
Preconditions (the caller guarantees these, so the loops are `@inbounds`):
`cell_hash[n] ∈ 0:total-1` for every particle `n`, where `total = length(cell_start)`;
`original_index` has length `N`; `cell_start`, `cell_end`, `counting_sort_scratch` all have
length `total`.

See `spec/particle-sorting.md`.
"""
function _build_cell_list_kernel!(
    original_index::Vector{Int32},
    cell_start::Vector{Int32},
    cell_end::Vector{Int32},
    counting_sort_scratch::Vector{Int32},
    cell_hash::Vector{Int32},
)
    total = length(cell_start)
    # counting_sort_scratch[c] counts the particles in cell c.
    fill!(counting_sort_scratch, Int32(0))
    @inbounds for n in eachindex(cell_hash)
        counting_sort_scratch[cell_hash[n] + Int32(1)] += Int32(1)
    end
    # Prefix sum into cell_start/cell_end; counting_sort_scratch becomes the
    # per-cell write cursor (the next free sorted slot of each cell).
    accumulator = Int32(1)
    @inbounds for c in 1:total
        cell_start[c] = accumulator
        accumulator += counting_sort_scratch[c]
        cell_end[c] = accumulator - Int32(1)
        counting_sort_scratch[c] = cell_start[c]
    end
    # Stable scatter: each particle lands at its cell's cursor.
    @inbounds for n in eachindex(cell_hash)
        c = cell_hash[n] + Int32(1)
        original_index[counting_sort_scratch[c]] = Int32(n)
        counting_sort_scratch[c] += Int32(1)
    end
    return original_index
end

"""
    sort_particles_by_cell!(config, Y, F) -> config

Step 2 of the Fast FCM algorithm (Su & Keaveny 2024, §4). Counting-sort the particles by
their cell hash and gather their positions and forces into that sorted order, so that
particles sharing a cell are contiguous in memory.

On return, `config` holds:

- `config.original_index` — sorted slot `s` to original particle index.
- `config.cell_start[c+1] : config.cell_end[c+1]` — 1-based inclusive range of
  sorted slots occupied by cell `c` (empty range if the cell is empty).
- `config.Y_sorted`, `config.F_sorted` — `Y` and `F` in sorted order.

# Arguments
- `config::FFCMConfig{T}`: the compiled configuration; the sort outputs are written into its
  buffers. Assumes `config.cell_hash` is current, i.e. `assign_cells!(config, Y)` ran since
  `Y` last changed.
- `Y::AbstractMatrix{T}`, `F::AbstractMatrix{T}`: the `3xN` positions and forces
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
        config.counting_sort_scratch,
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

Gather the `3xN` positions `Y` and forces `F` into sorted order under the permutation
`original_index`, writing `Y_sorted[:, s] = Y[:, original_index[s]]` and likewise for `F`,
so particles sharing a cell land in contiguous columns (the memory locality of the paper's
step 2).

# Arguments
- `Y_sorted::AbstractMatrix{T}`, `F_sorted::AbstractMatrix{T}`: the `3xN` outputs, written
  in sorted order.
- `Y::AbstractMatrix{T}`, `F::AbstractMatrix{T}`: the `3xN` source positions and forces in
  original order.
- `original_index::Vector{Int32}`: the sorted-slot to original-particle permutation.

# Returns
- `Y_sorted`: the same matrix, holding the gathered positions.

# Notes
Preconditions (caller-guaranteed, so the loop is `@inbounds`): all five arrays have second
dimension `N = length(original_index)`; `Y`, `F`, `Y_sorted`, `F_sorted` have first
dimension 3; `original_index[s] ∈ 1:N`.

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

Step 1 of the Fast FCM algorithm (Su & Keaveny 2024, §4). Write each particle's cell index
into `config.cell_hash`. The hash linearises the 3-D cell coordinate with `x` fastest and
`z` slowest.

# Arguments
- `config::FFCMConfig{T}`: the compiled configuration; `config.cell_hash` is overwritten.
- `Y::AbstractMatrix{T}`: the `3xN` positions, already folded into the canonical domain by
  `wrap_positions!`.

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

Kernel for `assign_cells!`: write `cell_hash[n] = x_c + (y_c + z_c ⋅ m_y) ⋅ m_x` for each
particle column of `Y`, with cell coordinates
`(x_c, y_c, z_c) = floor.(Y_n .* inv_cell_size)` and `num_cells[i] = m_i`.

# Arguments
- `cell_hash::Vector{Int32}`: length-`N` output; each particle's 0-based cell.
- `Y::AbstractMatrix{T}`: the `3xN` positions, folded into `[0, L_i)`.
- `inv_cell_size::NTuple{3, T}`: the inverse cell sizes.
- `num_cells::NTuple{3, Int32}`: the cell-grid dimensions `(m_x, m_y, m_z)`.

# Returns
- `cell_hash`: the same vector, holding the per-particle cell indices.

# Notes
`Y` is assumed already folded into `[0, L_i)` by `wrap_positions!`. The per-axis
`min(⋅, m_i - 1)` clamp defends against the floating point roundoff corner where
`Y_i ⋅ inv_cell_size_i` lands exactly at `m_i` and the floored cell index would otherwise be
one past the last valid cell.

See `spec/spatial-hashing.md`.
"""
function _assign_cells_kernel!(
    cell_hash::Vector{Int32},
    Y::AbstractMatrix{T},
    inv_cell_size::NTuple{3, T},
    num_cells::NTuple{3, Int32},
) where {T <: AbstractFloat}
    @inbounds @simd for n in axes(Y, 2)
        x_c = floor(Int32, Y[1, n] * inv_cell_size[1])
        y_c = floor(Int32, Y[2, n] * inv_cell_size[2])
        z_c = floor(Int32, Y[3, n] * inv_cell_size[3])
        x_c = min(x_c, num_cells[1] - Int32(1))
        y_c = min(y_c, num_cells[2] - Int32(1))
        z_c = min(z_c, num_cells[3] - Int32(1))
        cell_hash[n] = x_c + (y_c + z_c * num_cells[2]) * num_cells[1]
    end
    return cell_hash
end
