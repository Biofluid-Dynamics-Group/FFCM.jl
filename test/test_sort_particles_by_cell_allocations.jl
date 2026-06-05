using Test
using BenchmarkTools
using FFCM
using FFCM: sort_particles_by_cell!, _build_cell_list_kernel!,
    _gather_particles_kernel!, assign_cells!

# The hot path must allocate nothing: a single heap allocation per call would
# dominate the per-iteration cost in the downstream mobility solver.
@testset "Step-2 hot path does not allocate" begin
    for T in (Float32, Float64)
        N = 256
        L = (T(4), T(6), T(8))
        config = FFCMConfig{T}(;
            L = L, R_c = T(1), N = N, _fcm_grid_kwargs(L)...,
        )
        Y = rand(T, 3, N) .* T(4)
        F = rand(T, 3, N)
        assign_cells!(config, Y)
        @test (@ballocated sort_particles_by_cell!($config, $Y, $F)) == 0
        @test (@ballocated _build_cell_list_kernel!(
            $(config.original_index),
            $(config.cell_start),
            $(config.cell_end),
            $(config.next_free_slot),
            $(config.cell_hash),
        )) == 0
        @test (@ballocated _gather_particles_kernel!(
            $(config.Y_sorted),
            $(config.F_sorted),
            $Y,
            $F,
            $(config.original_index),
        )) == 0
    end
end
