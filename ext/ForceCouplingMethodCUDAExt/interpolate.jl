# GPU device kernel for pipeline step 5 — velocity interpolation / gather (Su & Keaveny
# 2024, §3 equation (26), §4). A device method of `_interpolate_velocities_kernel!`,
# dispatched on the device buffer types, so `interpolate_velocities!` and `mobility!` stay
# backend-agnostic. A block-per-particle shared-memory gather, monopole (force) only —
# the gather mirror of the step-3 spread's scatter, with no atomics (the gather only
# reads the grid and each particle writes its own output column).

# Block-per-particle launch: one block per particle (block-stride for N beyond the grid),
# 32 threads per block. The 32-thread block is a single warp, so the block reduction is a
# warp-shuffle reduction. Launch tuning is a deferred, benchmark-gated experiment.
@inline function _gather_launch(N::Integer)
    threads = 32
    blocks = max(Int(N), 1)
    return threads, blocks
end

# Dynamic shared-memory byte size: per-axis Gaussian weights and squared distances (`T`)
# and periodic-wrapped indices (`Int32`), `M_G` each, plus the staged particle position
# (3 `T`). No force is staged (the gather reads the grid, not F) and no reduction scratch
# is needed (the warp shuffle reduces in registers). The byte offsets in
# `_interpolate_velocities_device!` partition exactly this region.
@inline function _gather_shared_memory_bytes(::Type{T}, M_G::Integer) where {T}
    return Int(M_G) * (6 * sizeof(T) + 3 * sizeof(Int32)) + 3 * sizeof(T)
end

# Sum `val` across the 32 lanes of a single-warp block; lane 0 (threadIdx().x == 1) returns
# the total. All lanes must reach this call (full-mask shuffle), which the kernel guarantees
# by placing it after the uniform stencil sweep.
@inline function _warp_reduce_sum(val::T) where {T}
    offset = Int32(16)
    while offset > Int32(0)
        val += CUDA.shfl_down_sync(0xffffffff, val, offset)
        offset ÷= Int32(2)
    end
    return val
end

# One block gathers one particle's velocity. Threads first fill the separable per-axis
# stencil (Gaussian weight, squared distance, wrapped index) into shared memory — the
# step-3 spread precompute, duplicated here (deduping into a shared device helper is a
# deferred cleanup) — then each thread accumulates a partial modified-kernel-weighted
# velocity over its slice of the M_G³ support, the warp reduces the partials, and lane 0
# scales by h³ and writes the result to the particle's original-order column.
function _interpolate_velocities_device!(
    V, ux, uy, uz,
    Y_sorted, original_index,
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
    h³ = h^3

    # Partition the dynamic shared memory by byte offset (layout fixed by
    # `_gather_shared_memory_bytes`): six `T` axis arrays, then the staged position, then
    # three `Int32` index arrays.
    gauss_x = CuDynamicSharedArray(T, M_G)
    gauss_y = CuDynamicSharedArray(T, M_G, M_G * sizeof(T))
    gauss_z = CuDynamicSharedArray(T, M_G, 2 * M_G * sizeof(T))
    r²_x = CuDynamicSharedArray(T, M_G, 3 * M_G * sizeof(T))
    r²_y = CuDynamicSharedArray(T, M_G, 4 * M_G * sizeof(T))
    r²_z = CuDynamicSharedArray(T, M_G, 5 * M_G * sizeof(T))
    reals = 6 * M_G * sizeof(T)
    Y_priv = CuDynamicSharedArray(T, 3, reals)
    ints = reals + 3 * sizeof(T)
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

        # Each thread accumulates a partial gather over its slice of the M_G³ support patch;
        # threads beyond the patch contribute zero. The accumulator stays register-resident.
        v = zero(SVector{3, T})
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
                v += weight * SVector(ux[ix, iy, iz], uy[ix, iy, iz], uz[ix, iy, iz])
            end
            t += nthreads
        end

        # Warp-reduce the partials (single-warp block); lane 0 holds the total.
        vx = _warp_reduce_sum(v[1])
        vy = _warp_reduce_sum(v[2])
        vz = _warp_reduce_sum(v[3])
        if tid == Int32(1)
            @inbounds begin
                n = original_index[np]
                V[1, n] = h³ * vx
                V[2, n] = h³ * vy
                V[3, n] = h³ * vz
            end
        end
        sync_threads()

        np += gridDim().x
    end
    return nothing
end

function _interpolate_velocities_kernel!(
    V::CuMatrix{T},
    fluid_velocity,
    Y_sorted::CuMatrix{T},
    original_index::CuVector{Int32},
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
    ux, uy, uz = components(fluid_velocity)

    a₀, a₂, inv_norm, inv_2Σ² = _modified_kernel_coefficients(σ, Σ)

    N = size(Y_sorted, 2)
    threads, blocks = _gather_launch(N)
    shmem = _gather_shared_memory_bytes(T, M_G)
    @cuda threads = threads blocks = blocks shmem = shmem _interpolate_velocities_device!(
        V, ux, uy, uz,
        Y_sorted, original_index,
        a₀, a₂, inv_norm, inv_2Σ²,
        h, inv_h,
        num_grid_points, M_G, Int32(N),
    )
    return V
end
