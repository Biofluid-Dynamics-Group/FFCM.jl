using Test
using ForceCouplingMethod
using CUDA
using Random: Xoshiro, shuffle
using StructArrays: components
using ForceCouplingMethod: interpolate_velocities!

# CPU↔CUDA parity for pipeline step 5 (velocity interpolation / gather). Runs only where
# `CUDA.functional()` (guarded in runtests.jl). The GPU kernel is a block-per-particle
# shared-memory gather with a warp reduction over the M_G³ stencil, so the summation order
# differs from the CPU serial accumulation: the interpolated velocity matches the CPU
# backend to round-off, compared with the documented relative/absolute tolerances. The
# sorted positions, the sorted→original permutation, and the fluid velocity field are
# written directly and identically on both backends, so this isolates the gather from the
# cell-list sort and the upstream spread/solve.

@testset "GPU velocity interpolation matches the CPU backend" begin
    for T in (Float32, Float64)
        N = 256
        L = (T(8), T(8), T(8))
        rng = Xoshiro(0xC5A0 + (T === Float64 ? 1 : 0))

        cpu = _standard_test_config(T; N = N, L = L)
        gpu = _gpu_config(T; N = N, L = L)

        # In-box sorted positions and a fixed non-identity sorted→original permutation,
        # so the scatter-back through `original_index` is exercised, not just identity.
        Y_sorted = Matrix{T}(undef, 3, N)
        for n in 1:N, i in 1:3
            Y_sorted[i, n] = rand(rng, T) * L[i]
        end
        original_index = Int32.(shuffle(rng, collect(1:N)))
        copyto!(cpu.particles.Y_sorted, Y_sorted)
        copyto!(gpu.particles.Y_sorted, Y_sorted)
        copyto!(cpu.cells.original_index, original_index)
        copyto!(gpu.cells.original_index, original_index)

        # Random fluid velocity field, written straight into both backends' grids so the
        # gather is exercised without depending on the spread or the Stokes solve.
        grid_size = size(components(cpu.grid.fluid_velocity)[1])
        u = (randn(rng, T, grid_size), randn(rng, T, grid_size), randn(rng, T, grid_size))
        for (dest, src) in zip(components(cpu.grid.fluid_velocity), u)
            copyto!(dest, src)
        end
        for (dest, src) in zip(components(gpu.grid.fluid_velocity), u)
            copyto!(dest, src)
        end

        V_cpu = Matrix{T}(undef, 3, N)
        V_gpu = CuMatrix{T}(undef, 3, N)
        interpolate_velocities!(V_cpu, cpu)
        interpolate_velocities!(V_gpu, gpu)

        @test _cpu_gpu_isapprox(V_cpu, V_gpu; rtol = sqrt(eps(T)), atol = _near_zero_atol(T))
    end
end

@testset "GPU velocity interpolation is allocation-free" begin
    T = Float32
    N = 256
    L = (T(8), T(8), T(8))
    gpu = _gpu_config(T; N = N, L = L)
    rng = Xoshiro(0xC5A11)
    Y_sorted = Matrix{T}(undef, 3, N)
    for n in 1:N, i in 1:3
        Y_sorted[i, n] = rand(rng, T) * L[i]
    end
    copyto!(gpu.particles.Y_sorted, Y_sorted)
    copyto!(gpu.cells.original_index, Int32.(collect(1:N)))
    grid_size = size(components(gpu.grid.fluid_velocity)[1])
    for dest in components(gpu.grid.fluid_velocity)
        copyto!(dest, randn(rng, T, grid_size))
    end
    V_gpu = CuMatrix{T}(undef, 3, N)

    # Warm up the kernel (first launch JIT-compiles) before measuring.
    interpolate_velocities!(V_gpu, gpu)
    @test CUDA.@allocated(interpolate_velocities!(V_gpu, gpu)) == 0
end
