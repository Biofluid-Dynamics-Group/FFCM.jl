using Test
using FFCM
using CUDA
using LinearAlgebra: mul!
using StructArrays: components

# Runs only where `CUDA.functional()` is true (guarded in runtests.jl). C1
# establishes that a GPU-backed FFCMConfig *constructs* with device buffers and
# cuFFT plans; running `mobility!` on the GPU waits for the per-step kernels.

@testset "GPU-backed FFCMConfig construction" begin
    T = Float32
    L = (T(4), T(6), T(8))
    N = 16
    config = _standard_test_config(
        T; N = N, L = L, num_grid_points = (Int32(8), Int32(12), Int32(16)),
        M_G = 8, R_c = T(1), gpu_acceleration = true,
    )

    @testset "buffers are device-backed" begin
        @test config.cells.cell_hash isa CuArray
        @test config.cells.original_index isa CuArray
        @test config.particles.Y_sorted isa CuArray
        @test config.particles.Y_wrapped isa CuArray
        @test components(config.grid.force_density)[1] isa CuArray
        @test components(config.grid.fluid_velocity)[1] isa CuArray
        @test components(config.solver.fluid_hat)[1] isa CuArray
        @test config.solver.k_x isa CuArray
        @test components(config.stencil.stencil_gaussian)[1] isa CuArray
        @test components(config.stencil.stencil_index)[1] isa CuArray
    end

    @testset "derived scalars stay on the host" begin
        @test config.σ isa T
        @test config.μ isa T
        @test config.num_grid_points == (Int32(8), Int32(12), Int32(16))
        @test config.M_G == Int32(8)
    end

    @testset "cuFFT plans transform device arrays" begin
        force_x = components(config.grid.force_density)[1]
        hat_x = components(config.solver.fluid_hat)[1]
        mul!(hat_x, config.solver.forward_fourier_transform, force_x)
        @test hat_x isa CuArray
        @test size(hat_x) == (Int32(8) ÷ Int32(2) + Int32(1), Int32(12), Int32(16))
    end

    @testset "Float64 on the GPU warns but builds" begin
        config64 = @test_logs (:warn,) match_mode = :any _standard_test_config(
            Float64; N = N, L = (4.0, 6.0, 8.0),
            num_grid_points = (Int32(8), Int32(12), Int32(16)), M_G = 8, R_c = 1.0,
            gpu_acceleration = true,
        )
        @test config64.particles.Y_sorted isa CuArray
    end
end
