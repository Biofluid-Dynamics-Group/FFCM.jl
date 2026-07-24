using Test
using ForceCouplingMethod
using CUDA
using Random: Xoshiro
using StructArrays: components
using ForceCouplingMethod: stokes_solve!

# CPU↔CUDA parity for pipeline step 4 (the Fourier-space Stokes solve). Runs only where
# `CUDA.functional()` (guarded in runtests.jl). `stokes_solve!` is backend-agnostic: its
# forward/inverse transforms dispatch to cuFFT and its per-mode projection to a device
# method of `_apply_inverse_stokes_kernel!`. The projection is elementwise per Fourier mode
# (no atomics), so the solved velocity matches the CPU backend to round-off, compared with
# the documented relative/absolute tolerances. The force density is written directly and
# identically on both backends, so this isolates the solve from the spreading step.

@testset "GPU Stokes solve matches the CPU backend" begin
    for T in (Float32, Float64)
        N = 64
        L = (T(8), T(8), T(8))
        rng = Xoshiro(0xC4A0 + (T === Float64 ? 1 : 0))

        cpu = _standard_test_config(T; N = N, L = L)
        gpu = _gpu_config(T; N = N, L = L)

        # Random real force density, written straight into both backends' grids so the solve
        # is exercised without depending on the spreading step.
        grid_size = size(components(cpu.grid.force_density)[1])
        force = (randn(rng, T, grid_size), randn(rng, T, grid_size), randn(rng, T, grid_size))
        for (dest, src) in zip(components(cpu.grid.force_density), force)
            copyto!(dest, src)
        end
        for (dest, src) in zip(components(gpu.grid.force_density), force)
            copyto!(dest, src)
        end

        stokes_solve!(cpu)
        stokes_solve!(gpu)

        cux, cuy, cuz = components(cpu.grid.fluid_velocity)
        gux, guy, guz = components(gpu.grid.fluid_velocity)
        @test _cpu_gpu_isapprox(cux, gux; rtol = sqrt(eps(T)), atol = _near_zero_atol(T))
        @test _cpu_gpu_isapprox(cuy, guy; rtol = sqrt(eps(T)), atol = _near_zero_atol(T))
        @test _cpu_gpu_isapprox(cuz, guz; rtol = sqrt(eps(T)), atol = _near_zero_atol(T))
    end
end

@testset "GPU Stokes solve is allocation-free" begin
    T = Float32
    N = 64
    L = (T(8), T(8), T(8))
    gpu = _gpu_config(T; N = N, L = L)
    rng = Xoshiro(0xC4A11)
    grid_size = size(components(gpu.grid.force_density)[1])
    for dest in components(gpu.grid.force_density)
        copyto!(dest, randn(rng, T, grid_size))
    end

    # Warm up the kernel and cuFFT plans (first launch JIT-compiles) before measuring.
    stokes_solve!(gpu)
    @test CUDA.@allocated(stokes_solve!(gpu)) == 0
end
