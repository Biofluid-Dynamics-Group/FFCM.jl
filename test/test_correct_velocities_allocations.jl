using Test
using BenchmarkTools
using FFCM
using FFCM: correct_velocities!, _correct_velocities_kernel!,
    wrap_positions!, assign_cells!, sort_particles_by_cell!

# The hot path must allocate nothing: a single heap allocation per call would
# dominate the per-iteration cost in the downstream mobility solver.
@testset "Step-6 hot path does not allocate" begin
    for T in (Float32, Float64)
        L = (T(8), T(8), T(8))
        N = 32
        config = FFCMConfig{T}(;
            L = L, R_c = T(1), N = N,
            Σ_over_σ = T(2),
            μ = T(1),
            num_grid_points = (Int32(16), Int32(16), Int32(16)),
            M_G = 8,
        )
        # Cluster particles so the pair branch (including _correction_scalars) runs.
        Y = Matrix{T}(undef, 3, N)
        F = Matrix{T}(undef, 3, N)
        for n in 1:N
            Y[1, n] = T(4) + T(0.25) * ((n - 1) % 4)
            Y[2, n] = T(4) + T(0.25) * ((n - 1) ÷ 4 % 4)
            Y[3, n] = T(4) + T(0.25) * ((n - 1) ÷ 16)
            F[1, n] = T(0.1) * n
            F[2, n] = T(-0.2) * n
            F[3, n] = T(0.3) * n
        end
        wrap_positions!(Y, config.L)
        assign_cells!(config, Y)
        sort_particles_by_cell!(config, Y, F)
        V = zeros(T, 3, N)

        @test (@ballocated correct_velocities!($V, $config)) == 0
        @test (@ballocated _correct_velocities_kernel!(
            $V,
            $(config.Y_sorted),
            $(config.F_sorted),
            $(config.cell_start),
            $(config.cell_end),
            $(config.original_index),
            $(config.num_cells),
            $(config.L),
            $(config.σ),
            $(config.Σ),
            $(config.a),
            $(config.μ),
            $(config.R_c),
        )) == 0
    end
end
