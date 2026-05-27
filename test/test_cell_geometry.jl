using Test
using FFCM

@testset "Cell number computing matches the paper formula" begin
    config = FFCMConfig{Float64}(; L = (4.0, 6.0, 8.0), R_c = 1.0, N = 10)
    @test config.num_cells == (Int32(4), Int32(6), Int32(8))
end

@testset "Cell size and its inverse are consistent with L_i / m_i" begin
    config = FFCMConfig{Float64}(; L = (4.0, 6.0, 8.0), R_c = 1.0, N = 10)
    @test config.cell_size[1] ≈ 4.0 / config.num_cells[1]
    @test config.cell_size[2] ≈ 6.0 / config.num_cells[2]
    @test config.cell_size[3] ≈ 8.0 / config.num_cells[3]
    @test config.inv_cell_size[1] ≈ 1 / config.cell_size[1]
    @test config.inv_cell_size[2] ≈ 1 / config.cell_size[2]
    @test config.inv_cell_size[3] ≈ 1 / config.cell_size[3]
    @test all(config.cell_size .≥ config.R_c)
end

@testset "Tiny boxes are clamped to at least three cells per axis" begin
    # The paper's max(L_i / R_c, 3) floor guarantees the 26-neighbour stencil
    # covers every R_c-ball even when the box is too small for that to hold
    # naturally.
    config = FFCMConfig{Float64}(; L = (2.0, 2.0, 2.0), R_c = 1.0, N = 1)
    @test config.num_cells == (Int32(3), Int32(3), Int32(3))
end

@testset "Cell-list buffers are sized for the particle and cell counts" begin
    N = 10
    config = FFCMConfig{Float64}(; L = (4.0, 6.0, 8.0), R_c = 1.0, N = N)
    total = prod(Int(c) for c in config.num_cells)
    @test length(config.original_index) == N
    @test length(config.cell_start) == total
    @test length(config.cell_end) == total
    @test length(config.cell_cursor) == total
    @test size(config.Y_sorted) == (3, N)
    @test size(config.F_sorted) == (3, N)
end

@testset "Constructor rejects non-physical inputs" begin
    @test_throws ArgumentError FFCMConfig{Float64}(;
        L = (4.0, 4.0, 4.0), R_c = 0.0, N = 1,
    )
    @test_throws ArgumentError FFCMConfig{Float64}(;
        L = (4.0, 4.0, 4.0), R_c = -1.0, N = 1,
    )
    @test_throws ArgumentError FFCMConfig{Float64}(;
        L = (0.0, 4.0, 4.0), R_c = 1.0, N = 1,
    )
    @test_throws ArgumentError FFCMConfig{Float64}(;
        L = (4.0, 4.0, 4.0), R_c = 1.0, N = 0,
    )
end
