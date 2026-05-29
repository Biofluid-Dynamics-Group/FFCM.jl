using Test
using BenchmarkTools
using FFCM
using FFCM: interpolate_velocities!, _interpolate_velocities_kernel!

# The hot path must allocate nothing: a single heap allocation per call
# would dominate the per-iteration cost in the downstream mobility solver.
@testset "Step-5 hot path does not allocate" begin
    for T in (Float32, Float64)
        L = (T(8), T(8), T(8))
        N = 32
        config = FFCMConfig{T}(;
            L = L, R_c = T(1), N = N,
            Σ_over_σ = T(2),
            μ = T(1),
            num_grid_points = (Int32(8), Int32(8), Int32(8)),
            M_G = 8,
        )
        for n in 1:N
            for i in 1:3
                config.Y_sorted[i, n] = L[i] * T(0.5)
            end
            config.original_index[n] = Int32(n)
        end
        V = zeros(T, 3, N)
        @test (@ballocated interpolate_velocities!($V, $config)) == 0
        @test (@ballocated _interpolate_velocities_kernel!(
            $V,
            $(config.velocity_grid),
            $(config.Y_sorted),
            $(config.original_index),
            $(config.σ),
            $(config.Σ),
            $(config.Δx),
            $(config.inv_Δx),
            $(config.num_grid_points),
            $(config.M_G),
            $(config.gauss_x),
            $(config.gauss_y),
            $(config.gauss_z),
            $(config.r²_x),
            $(config.r²_y),
            $(config.r²_z),
            $(config.ind_x),
            $(config.ind_y),
            $(config.ind_z),
        )) == 0
    end
end
