"""
    wrap_positions!(Y, L) -> Y

Fold each column of `Y` (a `3×N` matrix of particle positions) into the
canonical periodic domain `[0, L_i)` for each axis `i`. Idempotent. The
fp-roundoff corner case where `mod(y, L_i)` rounds to exactly `L_i` is
left to the downstream cell-index clamp in `_assign_cells_kernel!`.

See `spec/spatial-hashing.md`.
"""
function wrap_positions!(Y::AbstractMatrix{T}, L::NTuple{3, T}) where {T}
    @inbounds @simd for n in axes(Y, 2)
        Y[1, n] = mod(Y[1, n], L[1])
        Y[2, n] = mod(Y[2, n], L[2])
        Y[3, n] = mod(Y[3, n], L[3])
    end
    return Y
end

"""
    assign_cells!(config, Y) -> config.cell_hash

Write each particle's cell index into `config.cell_hash`. Assumes `Y` has
already been folded into the canonical domain by `wrap_positions!`. The
hash linearises the 3-D cell coordinate with `x` fastest and `z` slowest,
matching paper §4 Step 1 and `cuFCM/src/CUFCM_CELLLIST.cu:create_hash_gpu`.

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
`cell_hash[n] = x_c + (y_c + z_c · m_y) · m_x` for each particle column
of `Y`, with cell coordinates `(x_c, y_c, z_c) = floor.(Y_n .* inv_cell_size)`
and `num_cells[i] = m_i`.

`Y` is assumed to have already been folded into `[0, L_i)` by
`wrap_positions!`. The per-axis `min(·, m_i - 1)` clamp defends against the
fp-roundoff corner where `Y_i · inv_cell_size_i` lands exactly at `m_i` and
the floored cell index would otherwise be one past the last valid cell.

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
