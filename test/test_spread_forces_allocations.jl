using Test
using BenchmarkTools
using FFCM
using FFCM: spread_forces!, _spread_forces_kernel!

# The hot path must allocate nothing: a single heap allocation per call
# would dominate the per-iteration cost in the downstream mobility solver.
@testset "Step-3 hot path does not allocate" begin
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
                config.F_sorted[i, n] = T(0.1) * T(n)
            end
        end
        @test (@ballocated spread_forces!($config)) == 0
        @test (@ballocated _spread_forces_kernel!(
            $(config.force_grid),
            $(config.Y_sorted),
            $(config.F_sorted),
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
