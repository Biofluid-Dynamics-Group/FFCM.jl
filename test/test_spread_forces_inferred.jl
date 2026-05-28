using Test
using FFCM
using FFCM: spread_forces!, _spread_forces_kernel!

# Type stability is load-bearing for the hot path: an abstract return type
# would force dynamic dispatch and break the allocation-free guarantee.
@testset "Step-3 entry points are type-stable for Float32 and Float64" begin
    for T in (Float32, Float64)
        L = (T(8), T(8), T(8))
        N = 8
        config = FFCMConfig{T}(;
            L = L, R_c = T(1), N = N,
            Σ_over_σ = T(2),
            num_grid_points = (Int32(8), Int32(8), Int32(8)),
            M_G = 8,
        )
        for n in 1:N
            for i in 1:3
                config.Y_sorted[i, n] = L[i] * T(0.5)
                config.F_sorted[i, n] = T(0.1) * T(n)
            end
        end
        @inferred spread_forces!(config)
        @inferred _spread_forces_kernel!(
            config.force_grid,
            config.Y_sorted,
            config.F_sorted,
            config.σ,
            config.Σ,
            config.Δx,
            config.inv_Δx,
            config.num_grid_points,
            config.M_G,
            config.gauss_x,
            config.gauss_y,
            config.gauss_z,
            config.r²_x,
            config.r²_y,
            config.r²_z,
            config.ind_x,
            config.ind_y,
            config.ind_z,
        )
    end
end
