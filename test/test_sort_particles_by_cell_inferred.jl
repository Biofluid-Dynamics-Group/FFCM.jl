using Test
using FFCM
using FFCM: sort_particles_by_cell!, _build_cell_list_kernel!,
    _gather_particles_kernel!, assign_cells!

# Type stability is load-bearing for the hot path: an abstract return type
# would force dynamic dispatch and break the allocation-free guarantee.
@testset "Step-2 entry points are type-stable for Float32 and Float64" begin
    for T in (Float32, Float64)
        N = 16
        L = (T(4), T(6), T(8))
        config = FFCMConfig{T}(;
            L = L, R_c = T(1), N = N, _fcm_grid_kwargs(L)...,
        )
        Y = rand(T, 3, N) .* T(4)
        F = rand(T, 3, N)
        assign_cells!(config, Y)
        @inferred sort_particles_by_cell!(config, Y, F)
        @inferred _build_cell_list_kernel!(
            config.original_index, config.cell_start, config.cell_end,
            config.cell_cursor, config.cell_hash,
        )
        @inferred _gather_particles_kernel!(
            config.Y_sorted, config.F_sorted, Y, F, config.original_index,
        )
    end
end
