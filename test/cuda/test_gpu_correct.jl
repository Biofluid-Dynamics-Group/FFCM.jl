using Test
using FFCM
using CUDA
using Random: Xoshiro
using FFCM: wrap_positions!, assign_cells!, sort_particles_by_cell!, correct_velocities!

# CPU↔CUDA parity for pipeline step 6 (the real-space pairwise correction). Runs only where
# `CUDA.functional()` (guarded in runtests.jl). The GPU kernel mirrors cuFCM: one thread per
# sorted particle, intra-cell self-gather plus a half-shell neighbour sweep with an atomic
# dual write, VF only, and the self term folded into the accumulator. The pair scalars are
# bit-identical to the CPU's (same `_correction_scalars` on the same separations), so only
# the atomic summation order differs — the corrected velocity matches the CPU backend to
# round-off, compared with the documented relative/absolute tolerances. The cell list is
# built once on the (tested) CPU backend and copied to the device, so this isolates the
# correction from the GPU cell-list sort. See spec/cuda-conventions.md,
# spec/pairwise-correction.md.

# Builds a consistent cell list on the CPU backend, then copies the sorted buffers and
# per-cell ranges onto the GPU config so both backends correct identical inputs. The device
# neighbour map is already resident from construction (geometry-only). Returns nothing;
# mutates `gpu`.
function _mirror_cpu_cell_list!(gpu, cpu, Y, F, L)
    wrap_positions!(cpu.particles.Y_wrapped, Y, L)
    assign_cells!(cpu, cpu.particles.Y_wrapped)
    sort_particles_by_cell!(cpu, cpu.particles.Y_wrapped, F)
    copyto!(gpu.particles.Y_sorted, cpu.particles.Y_sorted)
    copyto!(gpu.particles.F_sorted, cpu.particles.F_sorted)
    copyto!(gpu.cells.cell_start, cpu.cells.cell_start)
    copyto!(gpu.cells.cell_end, cpu.cells.cell_end)
    copyto!(gpu.cells.original_index, cpu.cells.original_index)
    return nothing
end

# A dense random cloud confined to a few cells around the box centre, so the sweep hits both
# intra-cell and inter-cell (half-shell) pairs within R_c and the atomic dual write is
# exercised; away from the boundary, so no pair needs a periodic wrap.
function _dense_centred_cloud(::Type{T}, N, L, rng) where {T}
    Y = Matrix{T}(undef, 3, N)
    F = Matrix{T}(undef, 3, N)
    for n in 1:N, i in 1:3
        Y[i, n] = L[i] / 2 - T(1.5) + T(3) * rand(rng, T)
        F[i, n] = rand(rng, T) - T(0.5)
    end
    return Y, F
end

@testset "GPU pairwise correction matches the CPU backend" begin
    for T in (Float32, Float64)
        N = 256
        L = (T(8), T(8), T(8))
        rng = Xoshiro(0xC6A0 + (T === Float64 ? 1 : 0))
        Y, F = _dense_centred_cloud(T, N, L, rng)

        cpu = _standard_test_config(T; N = N, L = L)
        gpu = _gpu_config(T; N = N, L = L)
        _mirror_cpu_cell_list!(gpu, cpu, Y, F, L)

        # Random pre-filled velocity (as if from interpolation): the correction is added on
        # top, so this checks the additive-onto-V contract as well as the pair sum.
        V0 = randn(rng, T, 3, N)
        V_cpu = copy(V0)
        V_gpu = CuArray(V0)
        correct_velocities!(V_cpu, cpu)
        correct_velocities!(V_gpu, gpu)

        @test _cpu_gpu_isapprox(V_cpu, V_gpu; rtol = sqrt(eps(T)), atol = _near_zero_atol(T))
    end
end

@testset "GPU pairwise correction vanishes at Σ = σ (matches the CPU backend)" begin
    T = Float64
    N = 128
    L = (T(8), T(8), T(8))
    rng = Xoshiro(0xC6A22)
    Y, F = _dense_centred_cloud(T, N, L, rng)

    cpu = _standard_test_config(T; N = N, L = L, Σ_over_σ = T(1))
    gpu = _gpu_config(T; N = N, L = L, Σ_over_σ = T(1))
    _mirror_cpu_cell_list!(gpu, cpu, Y, F, L)

    V0 = randn(rng, T, 3, N)
    V_cpu = copy(V0)
    V_gpu = CuArray(V0)
    correct_velocities!(V_cpu, cpu)
    correct_velocities!(V_gpu, gpu)

    # Correction is identically zero at Σ = σ, so both leave V0 untouched.
    @test _cpu_gpu_isapprox(V_cpu, V0; rtol = sqrt(eps(T)), atol = _near_zero_atol(T))
    @test _cpu_gpu_isapprox(V_gpu, V0; rtol = sqrt(eps(T)), atol = _near_zero_atol(T))
end

@testset "GPU pairwise correction is allocation-free" begin
    T = Float32
    N = 256
    L = (T(8), T(8), T(8))
    rng = Xoshiro(0xC6A11)
    Y, F = _dense_centred_cloud(T, N, L, rng)

    cpu = _standard_test_config(T; N = N, L = L)
    gpu = _gpu_config(T; N = N, L = L)
    _mirror_cpu_cell_list!(gpu, cpu, Y, F, L)
    V_gpu = CuArray(randn(rng, T, 3, N))

    # Warm up the kernel (first launch JIT-compiles) before measuring.
    correct_velocities!(V_gpu, gpu)
    @test CUDA.@allocated(correct_velocities!(V_gpu, gpu)) == 0
end
