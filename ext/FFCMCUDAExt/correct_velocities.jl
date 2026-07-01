# GPU device kernel for pipeline step 6 — the real-space pairwise correction (Su & Keaveny
# 2024, §3 equation (31), Appendix B, §4). A device method of `_correct_velocities_kernel!`,
# dispatched on the device buffer types (the `neighbor_map::CuVector` discriminates the GPU
# backend from the CPU's `neighbor_map::Nothing`), so `correct_velocities!` and `mobility!`
# stay backend-agnostic. The kernel mirrors cuFCM's active pair-correction kernel
# (`cufcm_pair_correction`), VF (force) only, with the self term folded into the
# per-particle accumulator (cuFCM applies it in a separate `cufcm_self_correction` kernel).
# See spec/pairwise-correction.md, spec/cuda-conventions.md.

# Grid-stride launch, one thread per sorted particle. 256 threads per block; launch tuning is
# a deferred, benchmark-gated experiment.
@inline function _correct_launch(N::Integer)
    threads = 256
    blocks = max(cld(Int(N), threads), 1)
    return threads, blocks
end

# One thread corrects one sorted particle `i`. It seeds its accumulator with the self term
# δF_i, sweeps its own cell (correction to `i` only — particle `j`'s thread corrects `j`),
# then sweeps the 13 forward half-shell neighbour cells, adding the pair correction to `i` in
# a register and to `j` with an atomic (Newton's-third-law dual write, the VF tensor being
# symmetric). The final self+to-`i` accumulation is atomic-added to `i`'s original-order
# column, atomic because a neighbour particle's thread may target the same column. `V` enters
# holding the interpolated velocity in original order, so every write is an `+=` correction.
function _correct_velocities_device!(
    V, Y_sorted, F_sorted,
    cell_start, cell_end, original_index, neighbor_map,
    num_cells::NTuple{3, Int32},
    inv_cell_size::NTuple{3, T},
    L::NTuple{3, T},
    self_correction_term::T,
    σ::T, Σ::T, μ::T, R_c²::T,
    N::Int32,
) where {T}
    m_x, m_y, m_z = num_cells

    index = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x
    i = index
    while i <= N
        @inbounds begin
            Yi = SVector{3, T}(Y_sorted[1, i], Y_sorted[2, i], Y_sorted[3, i])
            Fi = SVector{3, T}(F_sorted[1, i], F_sorted[2, i], F_sorted[3, i])
        end
        v = self_correction_term * Fi

        # Recompute particle i's cell from its (wrapped) position, the same floor/clamp/
        # linearise hash as `_assign_cells_device!`.
        x_c = min(floor(Int32, Yi[1] * inv_cell_size[1]), m_x - Int32(1))
        y_c = min(floor(Int32, Yi[2] * inv_cell_size[2]), m_y - Int32(1))
        z_c = min(floor(Int32, Yi[3] * inv_cell_size[3]), m_z - Int32(1))
        icell = x_c + (y_c + z_c * m_y) * m_x

        # Intra-cell: every other particle in i's cell, corrected onto i only.
        @inbounds for s in cell_start[icell + Int32(1)]:cell_end[icell + Int32(1)]
            s == i && continue
            Yj = SVector{3, T}(Y_sorted[1, s], Y_sorted[2, s], Y_sorted[3, s])
            x = _min_image(Yi - Yj, L)
            r² = dot(x, x)
            r² < R_c² || continue
            isotropic_coefficient, parallel_coefficient =
                _correction_scalars(sqrt(r²), σ, Σ, μ)
            Fj = SVector{3, T}(F_sorted[1, s], F_sorted[2, s], F_sorted[3, s])
            v += isotropic_coefficient * Fj + (parallel_coefficient * dot(x, Fj)) * x
        end

        # Inter-cell: the 13 forward half-shell neighbours. Each pair is visited once and
        # the symmetric correction is added to both particles (register for i, atomic for j).
        base = Int32(13) * icell
        for nabor in Int32(1):Int32(13)
            @inbounds jcell = neighbor_map[base + nabor]
            @inbounds for s in cell_start[jcell + Int32(1)]:cell_end[jcell + Int32(1)]
                Yj = SVector{3, T}(Y_sorted[1, s], Y_sorted[2, s], Y_sorted[3, s])
                x = _min_image(Yi - Yj, L)
                r² = dot(x, x)
                r² < R_c² || continue
                isotropic_coefficient, parallel_coefficient =
                    _correction_scalars(sqrt(r²), σ, Σ, μ)
                Fj = SVector{3, T}(F_sorted[1, s], F_sorted[2, s], F_sorted[3, s])
                v += isotropic_coefficient * Fj + (parallel_coefficient * dot(x, Fj)) * x

                # Correction to particle j from F_i (x → -x is antisymmetric, so the same
                # (isotropic, parallel) scalars and this expression give j's contribution).
                cj = isotropic_coefficient * Fi + (parallel_coefficient * dot(x, Fi)) * x
                @inbounds nj = original_index[s]
                CUDA.@atomic V[1, nj] += cj[1]
                CUDA.@atomic V[2, nj] += cj[2]
                CUDA.@atomic V[3, nj] += cj[3]
            end
        end

        @inbounds ni = original_index[i]
        CUDA.@atomic V[1, ni] += v[1]
        CUDA.@atomic V[2, ni] += v[2]
        CUDA.@atomic V[3, ni] += v[3]

        i += stride
    end
    return nothing
end

function _correct_velocities_kernel!(
    V::CuMatrix{T},
    Y_sorted::CuMatrix{T},
    F_sorted::CuMatrix{T},
    cell_start::CuVector{Int32},
    cell_end::CuVector{Int32},
    original_index::CuVector{Int32},
    neighbor_map::CuVector{Int32},
    num_cells::NTuple{3, Int32},
    inv_cell_size::NTuple{3, T},
    L::NTuple{3, T},
    σ::T,
    Σ::T,
    a::T,
    μ::T,
    R_c::T,
) where {T}
    self_correction_term = _self_correction(σ, Σ, a, μ)

    N = size(Y_sorted, 2)
    threads, blocks = _correct_launch(N)
    @cuda threads = threads blocks = blocks _correct_velocities_device!(
        V, Y_sorted, F_sorted,
        cell_start, cell_end, original_index, neighbor_map,
        num_cells, inv_cell_size, L,
        self_correction_term, σ, Σ, μ, R_c^2,
        Int32(N),
    )
    return V
end
