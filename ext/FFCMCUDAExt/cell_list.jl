# GPU device kernels for pipeline step 1 — spatial hashing and the cell list
# (Su & Keaveny 2024, §4). Device methods of the same step functions the CPU
# backend uses, dispatched on the device buffer types, so the public wrappers
# (`assign_cells!`, `sort_particles_by_cell!`) and `mobility!` are backend-agnostic.
# See spec/spatial-hashing.md, spec/particle-sorting.md, spec/cuda-conventions.md.

# Grid-stride launch shape, one thread per particle/slot. 256 threads per block
# is the default; launch tuning is a deferred, benchmark-gated experiment.
@inline function _step1_launch(n::Integer)
    threads = 256
    blocks = max(cld(Int(n), threads), 1)
    return threads, blocks
end

# --- device kernels ---------------------------------------------------------

# Fold each position into [0, L_i): the canonical periodic wrap.
function _wrap_positions_device!(dest, src, L, N)
    index = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x
    for n in index:stride:N
        @inbounds begin
            dest[1, n] = mod(src[1, n], L[1])
            dest[2, n] = mod(src[2, n], L[2])
            dest[3, n] = mod(src[3, n], L[3])
        end
    end
    return nothing
end

# Hash each wrapped position to its 0-based linear cell index (floor, clamp,
# linearise with x fastest), mirroring the CPU `_assign_cells_kernel!`.
function _assign_cells_device!(cell_hash, Y, inv_cell_size, num_cells, N)
    index = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x
    for n in index:stride:N
        @inbounds begin
            x_c = floor(Int32, Y[1, n] * inv_cell_size[1])
            y_c = floor(Int32, Y[2, n] * inv_cell_size[2])
            z_c = floor(Int32, Y[3, n] * inv_cell_size[3])
            x_c = min(x_c, num_cells[1] - Int32(1))
            y_c = min(y_c, num_cells[2] - Int32(1))
            z_c = min(z_c, num_cells[3] - Int32(1))
            cell_hash[n] = x_c + (y_c + z_c * num_cells[2]) * num_cells[1]
        end
    end
    return nothing
end

# Counting-sort pass 1: atomically tally the particles per cell.
function _histogram_cells_device!(counts, cell_hash, N)
    index = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x
    for i in index:stride:N
        @inbounds c = cell_hash[i]
        CUDA.@atomic counts[c + Int32(1)] += Int32(1)
    end
    return nothing
end

# Counting-sort pass 3: atomically claim each particle's slot from its cell's
# cursor and record the permutation. The atomic claim makes the scatter
# non-stable: intra-cell order is unspecified (spec/particle-sorting.md).
function _scatter_particles_device!(original_index, cursor, cell_hash, N)
    index = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x
    for i in index:stride:N
        @inbounds c = cell_hash[i]
        slot = CUDA.atomic_add!(pointer(cursor, c + Int32(1)), Int32(1))
        @inbounds original_index[slot] = Int32(i)
    end
    return nothing
end

# Gather positions and forces into cell-sorted order under the permutation.
function _gather_particles_device!(Y_sorted, F_sorted, Y, F, original_index, N)
    index = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x
    for s in index:stride:N
        @inbounds begin
            n = original_index[s]
            Y_sorted[1, s] = Y[1, n]
            Y_sorted[2, s] = Y[2, n]
            Y_sorted[3, s] = Y[3, n]
            F_sorted[1, s] = F[1, n]
            F_sorted[2, s] = F[2, n]
            F_sorted[3, s] = F[3, n]
        end
    end
    return nothing
end

# --- host-side step functions (device methods) ------------------------------

function wrap_positions!(
    dest::CuMatrix{T}, src::CuMatrix{T}, L::NTuple{3, T},
) where {T}
    N = size(src, 2)
    threads, blocks = _step1_launch(N)
    @cuda threads = threads blocks = blocks _wrap_positions_device!(dest, src, L, N)
    return dest
end

function _assign_cells_kernel!(
    cell_hash::CuVector{Int32},
    Y::CuMatrix{T},
    inv_cell_size::NTuple{3, T},
    num_cells::NTuple{3, Int32},
) where {T}
    N = size(Y, 2)
    threads, blocks = _step1_launch(N)
    @cuda threads = threads blocks = blocks _assign_cells_device!(
        cell_hash, Y, inv_cell_size, num_cells, N,
    )
    return cell_hash
end

# Parallel counting sort: histogram → prefix sum → atomic scatter, over the same
# buffers the CPU counting sort uses. The per-cell ranges follow from the prefix
# sum (empty cell ⇒ empty range); only the intra-cell order is non-stable.
function _build_cell_list_kernel!(
    original_index::CuVector{Int32},
    cell_start::CuVector{Int32},
    cell_end::CuVector{Int32},
    counting_sort_scratch::CuVector{Int32},
    cell_hash::CuVector{Int32},
)
    N = length(cell_hash)
    threads, blocks = _step1_launch(N)

    fill!(counting_sort_scratch, Int32(0))
    @cuda threads = threads blocks = blocks _histogram_cells_device!(
        counting_sort_scratch, cell_hash, N,
    )

    # Inclusive prefix sum of the counts gives each cell's last 1-based slot in
    # `cell_end`; the first slot follows from the per-cell count. `accumulate!` is
    # allocation-free for a single-block scan; a large cell count takes CUDA.jl's
    # multi-block scan, which allocates a transient buffer (spec/cuda-conventions.md).
    accumulate!(+, cell_end, counting_sort_scratch)
    cell_start .= cell_end .- counting_sort_scratch .+ Int32(1)

    # Reuse the scratch as the per-cell write cursor, seeded at each cell's first
    # slot; the scatter advances it atomically.
    copyto!(counting_sort_scratch, cell_start)
    @cuda threads = threads blocks = blocks _scatter_particles_device!(
        original_index, counting_sort_scratch, cell_hash, N,
    )
    return original_index
end

function _gather_particles_kernel!(
    Y_sorted::CuMatrix{T},
    F_sorted::CuMatrix{T},
    Y::CuMatrix{T},
    F::CuMatrix{T},
    original_index::CuVector{Int32},
) where {T}
    N = length(original_index)
    threads, blocks = _step1_launch(N)
    @cuda threads = threads blocks = blocks _gather_particles_device!(
        Y_sorted, F_sorted, Y, F, original_index, N,
    )
    return Y_sorted
end
