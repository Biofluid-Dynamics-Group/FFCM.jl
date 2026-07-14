using Test
using FFCM
using CUDA
using Random: Xoshiro
using StructArrays: components
using FFCM: spread_forces!

# CPU↔CUDA parity for pipeline step 3 (force spreading). Runs only where
# `CUDA.functional()` (guarded in runtests.jl). The GPU kernel is a block-per-particle
# shared-memory spread with an atomic scatter to the grid, so the accumulation order is
# unspecified: the spread density matches the CPU backend to round-off, compared with the
# documented relative/absolute tolerances. The sorted position/force buffers are populated
# directly and identically on both backends, so this isolates spreading from the
# (separately tested) cell-list sort.

@testset "GPU force spreading matches the CPU backend" begin
    for T in (Float32, Float64)
        N = 256
        L = (T(8), T(8), T(8))
        rng = Xoshiro(0xC3A0 + (T === Float64 ? 1 : 0))
        # In-box positions and random forces, written straight into the sorted buffers so
        # the spread is exercised without depending on the cell-list permutation.
        Y_sorted = Matrix{T}(undef, 3, N)
        F_sorted = Matrix{T}(undef, 3, N)
        for n in 1:N, i in 1:3
            Y_sorted[i, n] = rand(rng, T) * L[i]
            F_sorted[i, n] = rand(rng, T) - T(0.5)
        end

        cpu = _standard_test_config(T; N = N, L = L)
        gpu = _gpu_config(T; N = N, L = L)

        copyto!(cpu.particles.Y_sorted, Y_sorted)
        copyto!(cpu.particles.F_sorted, F_sorted)
        copyto!(gpu.particles.Y_sorted, Y_sorted)
        copyto!(gpu.particles.F_sorted, F_sorted)

        spread_forces!(cpu)
        spread_forces!(gpu)

        cfx, cfy, cfz = components(cpu.grid.force_density)
        gfx, gfy, gfz = components(gpu.grid.force_density)
        @test _cpu_gpu_isapprox(cfx, gfx; rtol = sqrt(eps(T)), atol = _near_zero_atol(T))
        @test _cpu_gpu_isapprox(cfy, gfy; rtol = sqrt(eps(T)), atol = _near_zero_atol(T))
        @test _cpu_gpu_isapprox(cfz, gfz; rtol = sqrt(eps(T)), atol = _near_zero_atol(T))
    end
end

@testset "GPU force spreading is allocation-free" begin
    T = Float32
    N = 256
    L = (T(8), T(8), T(8))
    gpu = _gpu_config(T; N = N, L = L)
    rng = Xoshiro(0xC3A11)
    Y_sorted = Matrix{T}(undef, 3, N)
    F_sorted = Matrix{T}(undef, 3, N)
    for n in 1:N, i in 1:3
        Y_sorted[i, n] = rand(rng, T) * L[i]
        F_sorted[i, n] = rand(rng, T) - T(0.5)
    end
    copyto!(gpu.particles.Y_sorted, Y_sorted)
    copyto!(gpu.particles.F_sorted, F_sorted)

    # Warm up the kernel (first launch JIT-compiles) before measuring.
    spread_forces!(gpu)
    @test CUDA.@allocated(spread_forces!(gpu)) == 0
end
