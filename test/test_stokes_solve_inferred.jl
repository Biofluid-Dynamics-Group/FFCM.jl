using Test
using FFCM
using FFCM: stokes_solve!, _apply_inverse_stokes_kernel!
using StructArrays: components

# Type stability is load-bearing for the hot path: an abstract return type
# would force dynamic dispatch and break the allocation-free guarantee.
@testset "Step-4 entry points are type-stable for Float32 and Float64" begin
    for T in (Float32, Float64)
        config = FFCMConfig{T}(;
            L = (T(4), T(4), T(4)), R_c = T(1), N = 1,
            Σ_over_σ = T(2),
            num_grid_points = (Int32(8), Int32(8), Int32(8)),
            M_G = 8, μ = T(1),
        )
        @inferred stokes_solve!(config)
        fx̂, fŷ, fẑ = components(config.fluid_hat)
        inv_M_total = one(T) / T(prod(Int, config.num_grid_points))
        @inferred _apply_inverse_stokes_kernel!(
            fx̂, fŷ, fẑ,
            config.k_x, config.k_y, config.k_z,
            config.μ, inv_M_total,
        )
    end
end
