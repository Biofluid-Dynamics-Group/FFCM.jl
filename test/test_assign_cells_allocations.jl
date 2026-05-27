using Test
using BenchmarkTools
using FFCM
using FFCM: assign_cells!, _assign_cells_kernel!, wrap_positions!

# The hot path must allocate nothing. Any heap allocation inside the loop
# would dominate the per-call cost in the iterative mobility solver the
# package is designed for.
@testset "Hot-path passes do not allocate" begin
    for T in (Float32, Float64)
        L = (T(4), T(6), T(8))
        config = FFCMConfig{T}(; L = L, R_c = T(1), N = 256)
        Y = rand(T, 3, 256) .* T(4)
        @test (@ballocated wrap_positions!($Y, $(config.L))) == 0
        @test (@ballocated assign_cells!($config, $Y)) == 0
        @test (@ballocated _assign_cells_kernel!(
            $(config.cell_hash),
            $Y,
            $(config.inv_cell_size),
            $(config.num_cells),
        )) == 0
    end
end
