using Test
using BenchmarkTools
using FFCM
using FFCM: stokes_solve!, _apply_inverse_stokes_kernel!
using StructArrays: components

# The hot path must allocate nothing: a single heap allocation per call
# would dominate the per-iteration cost in the downstream mobility solver.
@testset "Step-4 hot path does not allocate" begin
    for T in (Float32, Float64)
        config = FFCMConfig{T}(;
            L = (T(4), T(4), T(4)), R_c = T(1), N = 1,
            Σ_over_σ = T(2),
            num_grid_points = (Int32(8), Int32(8), Int32(8)),
            M_G = 8, μ = T(1),
        )
        @test (@ballocated stokes_solve!($config)) == 0

        fx̂, fŷ, fẑ = components(config.fluid_hat)
        inv_M_total = one(T) / T(prod(Int, config.num_grid_points))
        @test (@ballocated _apply_inverse_stokes_kernel!(
            $fx̂, $fŷ, $fẑ,
            $(config.k_x), $(config.k_y), $(config.k_z),
            $(config.μ), $inv_M_total,
        )) == 0
    end
end
