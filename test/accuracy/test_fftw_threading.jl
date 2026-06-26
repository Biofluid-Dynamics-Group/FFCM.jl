using Test
using FFCM
using FFCM: stokes_solve!

@testset "Threaded Stokes solve matches the single-thread reference" begin
    # FFTW executes a threaded plan by partitioning the transform across
    # tasks, which reorders floating-point summation but changes nothing
    # else — threaded and serial solves agree to round-off. The comparison
    # runs the same particles through a serial-plan and a threaded-plan
    # config, at the step level (the velocity grid out of the same spread
    # forces) and end-to-end (mobility! velocities). Velocity fields mix
    # large and near-zero entries, hence the combined rtol/atol.
    for T in (Float32, Float64)
        N = 32
        config_serial = _standard_test_config(T; N = N)
        config_threaded = _standard_test_config(T; N = N, fft_threads = 2)
        Y, F = _clustered_cloud(T, N)

        V_serial = zeros(T, 3, N)
        V_threaded = zeros(T, 3, N)
        mobility!(V_serial, config_serial, Y, F)
        mobility!(V_threaded, config_threaded, Y, F)
        @test all(isapprox.(
            V_threaded, V_serial;
            rtol = sqrt(eps(T)), atol = _near_zero_atol(T),
        ))

        # Step-level parity: mobility! leaves the spread forces in place, so
        # re-running the solve compares the velocity grids on identical input.
        stokes_solve!(config_serial)
        stokes_solve!(config_threaded)
        @test all(isapprox.(
            config_threaded.grid.fluid_velocity, config_serial.grid.fluid_velocity;
            rtol = sqrt(eps(T)), atol = _near_zero_atol(T),
        ))
    end
end

@testset "Non-positive FFT thread count is rejected at construction" begin
    for T in (Float32, Float64)
        @test_throws ArgumentError FFCMConfig{T}(;
            L = (T(8), T(8), T(8)), R_c = T(1), N = 1, kernel_widths_ratio = T(2),
            viscosity = T(1), num_grid_points = (Int32(16), Int32(16), Int32(16)),
            M_G = 8, fft_threads = 0,
        )
    end
end
