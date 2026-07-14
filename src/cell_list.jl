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
"""
function wrap_positions!(
    destination::AbstractMatrix{T}, source::AbstractMatrix{T}, L::NTuple{3, T},
) where {T}
    # `L` broadcasts as a length-3 collection along the axis dimension.
    destination .= mod.(source, L)
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

- `config.cells.original_index` — sorted slot `s` to original particle index.
- `config.cells.cell_start[c+1] : config.cells.cell_end[c+1]` — 1-based inclusive range of
  sorted slots occupied by cell `c` (empty range if the cell is empty).
- `config.particles.Y_sorted`, `config.particles.F_sorted` — `Y` and `F` in sorted order.

# Arguments
- `config::FFCMConfig{T}`: the compiled configuration; the sort outputs are written into its
  buffers. Assumes `config.cells.cell_hash` is current, i.e. `assign_cells!(config, Y)` ran since
  `Y` last changed.
- `Y::AbstractMatrix{T}`, `F::AbstractMatrix{T}`: the `3xN` positions and forces
  to gather into sorted order.

# Returns
- `config`: the same configuration, with the sort outputs populated.
"""
function sort_particles_by_cell!(
    config::FFCMConfig{T}, Y::AbstractMatrix{T}, F::AbstractMatrix{T},
) where {T}
    _build_cell_list_kernel!(
        config.cells.original_index,
        config.cells.cell_start,
        config.cells.cell_end,
        config.cells.counting_sort_scratch,
        config.cells.cell_hash,
    )
    _gather_particles_kernel!(
        config.particles.Y_sorted, config.particles.F_sorted, Y, F, config.cells.original_index,
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
    assign_cells!(config, Y) -> config.cells.cell_hash

Step 1 of the Fast FCM algorithm (Su & Keaveny 2024, §4). Write each particle's cell index
into `config.cells.cell_hash`. The hash linearises the 3-D cell coordinate with `x` fastest and
`z` slowest.

# Arguments
- `config::FFCMConfig{T}`: the compiled configuration; `config.cells.cell_hash` is overwritten.
- `Y::AbstractMatrix{T}`: the `3xN` positions, already folded into the canonical domain by
  `wrap_positions!`.

# Returns
- `config.cells.cell_hash`: the per-particle 0-based cell indices.
"""
function assign_cells!(config::FFCMConfig{T}, Y::AbstractMatrix{T}) where {T}
    _assign_cells_kernel!(
        config.cells.cell_hash, Y, config.inv_cell_size, config.num_cells,
    )
    return config.cells.cell_hash
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

"""
    _build_neighbor_map(num_cells) -> Vector{Int32}

Build the half-shell neighbour map for the GPU pairwise correction (paper §4): for each cell
`c ∈ 0:total-1`, the 0-based linear indices of its 13 forward neighbour cells, laid out
contiguously so cell `c` owns slots `13c+1 : 13c+13`. The 13 offsets are one half of the 26
surrounding cells, chosen so each unordered neighbouring pair of cells is listed once — the
GPU correction adds each pairwise term to both particles, so only half the shell is walked.

Geometry-only and built once on the host; the CUDA extension copies the result to the device.

# Arguments
- `num_cells::NTuple{3, Int32}`: the cell-grid dimensions `(m_x, m_y, m_z)`, each `≥ 3`, so
  the periodic wrap maps the 13 offsets to cells distinct from `c`.

# Returns
- `Vector{Int32}` of length `13 * prod(num_cells)`: the per-cell neighbour cell indices.
"""
function _build_neighbor_map(num_cells::NTuple{3, Int32})
    m_x, m_y, m_z = num_cells
    offsets = (
        (Int32(1), Int32(0), Int32(0)),
        (Int32(1), Int32(1), Int32(0)),
        (Int32(0), Int32(1), Int32(0)),
        (Int32(-1), Int32(1), Int32(0)),
        (Int32(1), Int32(0), Int32(-1)),
        (Int32(1), Int32(1), Int32(-1)),
        (Int32(0), Int32(1), Int32(-1)),
        (Int32(-1), Int32(1), Int32(-1)),
        (Int32(1), Int32(0), Int32(1)),
        (Int32(1), Int32(1), Int32(1)),
        (Int32(0), Int32(1), Int32(1)),
        (Int32(-1), Int32(1), Int32(1)),
        (Int32(0), Int32(0), Int32(1)),
    )
    total = Int(m_x) * Int(m_y) * Int(m_z)
    neighbor_map = Vector{Int32}(undef, 13 * total)
    for cz in Int32(0):(m_z - Int32(1)), cy in Int32(0):(m_y - Int32(1)),
        cx in Int32(0):(m_x - Int32(1))

        base = 13 * Int(cx + (cy + cz * m_y) * m_x)
        for (k, (dx, dy, dz)) in enumerate(offsets)
            nx = mod(cx + dx, m_x)
            ny = mod(cy + dy, m_y)
            nz = mod(cz + dz, m_z)
            neighbor_map[base + k] = nx + (ny + nz * m_y) * m_x
        end
    end
    return neighbor_map
end
