using Test
using FFCM
using FFCM: interpolate_velocities!, _interpolate_velocities_kernel!

# Type stability is load-bearing for the hot path: an abstract return type
# would force dynamic dispatch and break the allocation-free guarantee.
@testset "Step-5 entry points are type-stable for Float32 and Float64" begin
    for T in (Float32, Float64)
        L = (T(8), T(8), T(8))
        N = 8
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
        @inferred interpolate_velocities!(V, config)
        @inferred _interpolate_velocities_kernel!(
            V,
            config.fluid_velocity,
            config.Y_sorted,
            config.original_index,
            config.σ,
            config.Σ,
            config.h,
            config.inv_h,
            config.num_grid_points,
            config.M_G,
            config.gaussian_x,
            config.gaussian_y,
            config.gaussian_z,
            config.r²_x,
            config.r²_y,
            config.r²_z,
            config.idx_x,
            config.idx_y,
            config.idx_z,
        )
    end
end
