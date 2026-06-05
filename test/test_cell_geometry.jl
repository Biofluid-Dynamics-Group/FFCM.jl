using Test
using FFCM

@testset "Cell number computing matches the paper formula" begin
    L = (4.0, 6.0, 8.0)
    config = FFCMConfig{Float64}(;
        L = L, R_c = 1.0, N = 10, _fcm_grid_kwargs(L)...,
    )
    @test config.num_cells == (Int32(4), Int32(6), Int32(8))
end

@testset "Cell size and its inverse are consistent with L_i / m_i" begin
    L = (4.0, 6.0, 8.0)
    config = FFCMConfig{Float64}(;
        L = L, R_c = 1.0, N = 10, _fcm_grid_kwargs(L)...,
    )
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
    L = (2.0, 2.0, 2.0)
    config = FFCMConfig{Float64}(;
        L = L, R_c = 1.0, N = 1, _fcm_grid_kwargs(L)...,
    )
    @test config.num_cells == (Int32(3), Int32(3), Int32(3))
end

@testset "Cell-list buffers are sized for the particle and cell counts" begin
    N = 10
    L = (4.0, 6.0, 8.0)
    config = FFCMConfig{Float64}(;
        L = L, R_c = 1.0, N = N, _fcm_grid_kwargs(L)...,
    )
    total = prod(Int(c) for c in config.num_cells)
    @test length(config.original_index) == N
    @test length(config.cell_start) == total
    @test length(config.cell_end) == total
    @test length(config.next_free_slot) == total
    @test size(config.Y_sorted) == (3, N)
    @test size(config.F_sorted) == (3, N)
end

@testset "Constructor rejects non-physical inputs" begin
    L_ok = (4.0, 4.0, 4.0)
    kw = _fcm_grid_kwargs(L_ok)
    @test_throws ArgumentError FFCMConfig{Float64}(;
        L = L_ok, R_c = 0.0, N = 1, kw...,
    )
    @test_throws ArgumentError FFCMConfig{Float64}(;
        L = L_ok, R_c = -1.0, N = 1, kw...,
    )
    # L_x = 0 with the matching `num_grid_points` would divide by zero in
    # the spreading kwargs; we keep the helper sized for L_ok and let the
    # constructor reject the zero L component first.
    @test_throws ArgumentError FFCMConfig{Float64}(;
        L = (0.0, 4.0, 4.0), R_c = 1.0, N = 1, kw...,
    )
    @test_throws ArgumentError FFCMConfig{Float64}(;
        L = L_ok, R_c = 1.0, N = 0, kw...,
    )
    # R_c above half the smallest box length makes the minimum image ambiguous
    # and would allow a particle to correct against its own image
    # (paper outline.tex:318); the constructor rejects it. R_c = min(L)/2 is the
    # boundary and is allowed.
    L_thin = (4.0, 4.0, 2.0)
    kw_thin = _fcm_grid_kwargs(L_thin)
    @test_throws ArgumentError FFCMConfig{Float64}(;
        L = L_thin, R_c = 1.5, N = 1, kw_thin...,
    )
    @test FFCMConfig{Float64}(;
        L = L_thin, R_c = 1.0, N = 1, kw_thin...,
    ) isa FFCMConfig
end
