using Test
using FFCM
using FFCM: assign_cells!, _assign_cells_kernel!, wrap_positions!

# Type stability is a load-bearing invariant for the hot path: any abstract
# return value would force a dynamic dispatch inside the per-particle loop
# and break the allocation-free guarantee.
@testset "Hot-path entry points are type-stable for Float32 and Float64" begin
    for T in (Float32, Float64)
        L = (T(4), T(6), T(8))
        config = FFCMConfig{T}(;
            L = L, R_c = T(1), N = 16, _fcm_grid_kwargs(L)...,
        )
        Y = rand(T, 3, 16) .* T(4)
        @inferred wrap_positions!(Y, config.L)
        @inferred assign_cells!(config, Y)
        @inferred _assign_cells_kernel!(
            config.cell_hash, Y, config.inv_cell_size, config.num_cells,
        )
    end
end
