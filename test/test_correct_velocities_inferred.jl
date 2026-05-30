using Test
using FFCM
using FFCM: correct_velocities!, _correct_velocities_kernel!,
    _correction_scalars, _self_correction,
    wrap_positions!, assign_cells!, sort_particles_by_cell!

# Type stability is load-bearing for the hot path: an abstract return type
# would force dynamic dispatch and break the allocation-free guarantee.
@testset "Step-6 entry points are type-stable for Float32 and Float64" begin
    for T in (Float32, Float64)
        L = (T(8), T(8), T(8))
        N = 8
        config = FFCMConfig{T}(;
            L = L, R_c = T(1), N = N,
            Σ_over_σ = T(2),
            μ = T(1),
            num_grid_points = (Int32(16), Int32(16), Int32(16)),
            M_G = 8,
        )
        # Cluster the particles so several pairs fall within R_c and the pair
        # branch is exercised; populate the cell list via steps 1-2.
        Y = Matrix{T}(undef, 3, N)
        F = Matrix{T}(undef, 3, N)
        for n in 1:N
            Y[1, n] = T(4) + T(0.3) * (n - 1)
            Y[2, n] = T(4)
            Y[3, n] = T(4)
            F[1, n] = T(0.1) * n
            F[2, n] = T(-0.2) * n
            F[3, n] = T(0.3) * n
        end
        wrap_positions!(Y, config.L)
        assign_cells!(config, Y)
        sort_particles_by_cell!(config, Y, F)
        V = zeros(T, 3, N)

        @inferred _self_correction(config.σ, config.Σ, config.a, config.μ)
        @inferred _correction_scalars(T(0.5), config.σ, config.Σ, config.μ)
        @inferred correct_velocities!(V, config)
        @inferred _correct_velocities_kernel!(
            V,
            config.Y_sorted,
            config.F_sorted,
            config.cell_start,
            config.cell_end,
            config.original_index,
            config.num_cells,
            config.L,
            config.σ,
            config.Σ,
            config.a,
            config.μ,
            config.R_c,
        )
    end
end
