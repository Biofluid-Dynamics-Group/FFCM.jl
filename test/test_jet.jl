using Test
using JET
using FFCM
using FFCM: assign_cells!, _assign_cells_kernel!, wrap_positions!,
    sort_particles_by_cell!

# JET catches dispatch / inference problems that `@inferred` alone can miss
# (e.g., method-instability in callees, abstract field accesses) by walking
# the full call graph from each entry point.
@testset "JET: inference health for hot-path entry points" begin
    for T in (Float32, Float64)
        config = FFCMConfig{T}(;
            L = (T(4), T(6), T(8)),
            R_c = T(1),
            N = 16,
        )
        Y = rand(T, 3, 16) .* T(4)
        F = rand(T, 3, 16)
        @test_call wrap_positions!(Y, config.L)
        @test_call assign_cells!(config, Y)
        @test_call _assign_cells_kernel!(
            config.cell_hash, Y, config.inv_cell_size, config.num_cells,
        )
        @test_call sort_particles_by_cell!(config, Y, F)
    end
end
