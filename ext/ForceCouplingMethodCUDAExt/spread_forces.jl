# GPU device kernel for pipeline step 3 — force spreading (Su & Keaveny 2024, §3
# equation (25), §4). A device method of `_spread_forces_kernel!`, dispatched on the
# device buffer types, so `spread_forces!` and `mobility!` stay backend-agnostic. A
# block-per-particle kernel, monopole (force) only: each block stages its particle's
# separable stencil in dynamic shared memory and atomically scatters the force to the
# grid.

# Block-per-particle launch: one block per particle (block-stride for N beyond the grid),
# 32 threads per block (one warp). Launch tuning is a deferred, benchmark-gated
# experiment.
@inline function _spread_launch(N::Integer)
    threads = 32
    blocks = max(Int(N), 1)
    return threads, blocks
end

# Dynamic shared-memory byte size: per-axis Gaussian weights and squared distances (`T`)
# and periodic-wrapped indices (`Int32`), `M_G` each, plus the staged particle position
# and force (3 `T` each). The byte offsets in `_spread_forces_device!` partition exactly
# this region.
@inline function _spread_shmem_bytes(::Type{T}, M_G::Integer) where {T}
    return Int(M_G) * (6 * sizeof(T) + 3 * sizeof(Int32)) + 6 * sizeof(T)
end

# One block spreads one particle's force. Threads first fill the separable per-axis stencil
# (Gaussian weight, squared distance, wrapped index) into shared memory, then sweep the
# M_G³ support patch and atomically add the modified-kernel-weighted force to the grid.
function _spread_forces_device!(
    fx, fy, fz,
    Y_sorted, F_sorted,
    a₀::T, a₂::T, inv_norm::T, inv_2Σ²::T,
    h::T, inv_h::T,
    num_grid_points::NTuple{3, Int32},
    M_G::Int32,
    N::Int32,
) where {T}
    M_x, M_y, M_z = num_grid_points
    half_M_G = M_G ÷ Int32(2)
    M_G² = M_G * M_G
    patch = M_G² * M_G

    # Partition the dynamic shared memory by byte offset (layout fixed by
    # `_spread_shmem_bytes`): six `T` axis arrays, then the staged position and force,
    # then three `Int32` index arrays.
    gauss_x = CuDynamicSharedArray(T, M_G)
    gauss_y = CuDynamicSharedArray(T, M_G, M_G * sizeof(T))
    gauss_z = CuDynamicSharedArray(T, M_G, 2 * M_G * sizeof(T))
    r²_x = CuDynamicSharedArray(T, M_G, 3 * M_G * sizeof(T))
    r²_y = CuDynamicSharedArray(T, M_G, 4 * M_G * sizeof(T))
    r²_z = CuDynamicSharedArray(T, M_G, 5 * M_G * sizeof(T))
    reals = 6 * M_G * sizeof(T)
    Y_priv = CuDynamicSharedArray(T, 3, reals)
    F_priv = CuDynamicSharedArray(T, 3, reals + 3 * sizeof(T))
    ints = reals + 6 * sizeof(T)
    idx_x = CuDynamicSharedArray(Int32, M_G, ints)
    idx_y = CuDynamicSharedArray(Int32, M_G, ints + M_G * sizeof(Int32))
    idx_z = CuDynamicSharedArray(Int32, M_G, ints + 2 * M_G * sizeof(Int32))

    tid = threadIdx().x
    nthreads = blockDim().x

    np = blockIdx().x
    while np <= N
        if tid == Int32(1)
            @inbounds begin
                Y_priv[1] = Y_sorted[1, np]
                Y_priv[2] = Y_sorted[2, np]
                Y_priv[3] = Y_sorted[3, np]
                F_priv[1] = F_sorted[1, np]
                F_priv[2] = F_sorted[2, np]
                F_priv[3] = F_sorted[3, np]
            end
        end
        sync_threads()

        @inbounds Y1, Y2, Y3 = Y_priv[1], Y_priv[2], Y_priv[3]
        # Nearest-grid anchor per axis (the anchoring the paper's §5 Table 1 calibration
        # assumes), matching the CPU `_fill_particle_stencil!`.
        j1 = round(Int32, Y1 * inv_h)
        j2 = round(Int32, Y2 * inv_h)
        j3 = round(Int32, Y3 * inv_h)

        # Separable per-axis precompute into shared memory.
        k = tid
        while k <= M_G
            offset = k - Int32(1) - half_M_G
            g1 = j1 + offset
            g2 = j2 + offset
            g3 = j3 + offset
            x1 = T(g1) * h - Y1
            x2 = T(g2) * h - Y2
            x3 = T(g3) * h - Y3
            x1² = x1 * x1
            x2² = x2 * x2
            x3² = x3 * x3
            @inbounds begin
                gauss_x[k] = inv_norm * exp(-x1² * inv_2Σ²)
                gauss_y[k] = inv_norm * exp(-x2² * inv_2Σ²)
                gauss_z[k] = inv_norm * exp(-x3² * inv_2Σ²)
                r²_x[k] = x1²
                r²_y[k] = x2²
                r²_z[k] = x3²
                idx_x[k] = mod(g1, M_x) + Int32(1)
                idx_y[k] = mod(g2, M_y) + Int32(1)
                idx_z[k] = mod(g3, M_z) + Int32(1)
            end
            k += nthreads
        end
        sync_threads()

        @inbounds F1, F2, F3 = F_priv[1], F_priv[2], F_priv[3]

        # Atomic scatter over the M_G³ support patch (column-major linear grid index).
        t = tid
        while t <= patch
            t0 = t - Int32(1)
            kz = t0 ÷ M_G² + Int32(1)
            rest = t0 - (kz - Int32(1)) * M_G²
            ky = rest ÷ M_G + Int32(1)
            kx = rest - (ky - Int32(1)) * M_G + Int32(1)
            @inbounds begin
                ix, iy, iz = idx_x[kx], idx_y[ky], idx_z[kz]
                r² = r²_x[kx] + (r²_y[ky] + r²_z[kz])
                weight = (a₀ + a₂ * r²) * gauss_x[kx] * (gauss_y[ky] * gauss_z[kz])
                lin = ix + (iy - Int32(1)) * M_x + (iz - Int32(1)) * M_x * M_y
                CUDA.@atomic fx[lin] += F1 * weight
                CUDA.@atomic fy[lin] += F2 * weight
                CUDA.@atomic fz[lin] += F3 * weight
            end
            t += nthreads
        end
        sync_threads()

        np += gridDim().x
    end
    return nothing
end

function _spread_forces_kernel!(
    force_density,
    Y_sorted::CuMatrix{T},
    F_sorted::CuMatrix{T},
    σ::T,
    Σ::T,
    h::T,
    inv_h::T,
    num_grid_points::NTuple{3, Int32},
    M_G::Int32,
    stencil_gaussian,
    stencil_r²,
    stencil_index,
) where {T}
    fx, fy, fz = components(force_density)
    fill!(fx, zero(T))
    fill!(fy, zero(T))
    fill!(fz, zero(T))

    a₀, a₂, inv_norm, inv_2Σ² = _modified_kernel_coefficients(σ, Σ)

    N = size(Y_sorted, 2)
    threads, blocks = _spread_launch(N)
    shmem = _spread_shmem_bytes(T, M_G)
    @cuda threads = threads blocks = blocks shmem = shmem _spread_forces_device!(
        fx, fy, fz,
        Y_sorted, F_sorted,
        a₀, a₂, inv_norm, inv_2Σ²,
        h, inv_h,
        num_grid_points, M_G, Int32(N),
    )
    return force_density
end
