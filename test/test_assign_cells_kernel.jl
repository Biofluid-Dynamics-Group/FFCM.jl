using Test
using FFCM: _assign_cells_kernel!

@testset "Hash for a position in the origin cell is zero" begin
    # Cubic 4×4×4 grid: cell_size_i = 1, inv_cell_size_i = 1, m_i = 4.
    cell_hash = Vector{Int32}(undef, 1)
    inv_cell_size = (1.0, 1.0, 1.0)
    num_cells = (Int32(4), Int32(4), Int32(4))
    Y = Float64[0.5; 0.5; 0.5;;]
    _assign_cells_kernel!(cell_hash, Y, inv_cell_size, num_cells)
    @test cell_hash[1] == Int32(0)
end

@testset "Hash matches the paper linearisation x + (y + z·m_y)·m_x" begin
    # Particle at (2.5, 2.5, 2.5) in a 4×4×4 grid sits in cell (2, 2, 2).
    # Expected hash: 2 + (2 + 2·4)·4 = 42.
    cell_hash = Vector{Int32}(undef, 1)
    inv_cell_size = (1.0, 1.0, 1.0)
    num_cells = (Int32(4), Int32(4), Int32(4))
    Y = Float64[2.5; 2.5; 2.5;;]
    _assign_cells_kernel!(cell_hash, Y, inv_cell_size, num_cells)
    @test cell_hash[1] == Int32(42)
end

@testset "Anisotropic grid uses m_x and m_y as the linearisation strides" begin
    # L = (4, 6, 8), R_c = 1 ⇒ num_cells = (4, 6, 8), cell_size_i = 1.
    # Particle at (0.5, 2.5, 5.5) sits in cell (0, 2, 5).
    # Expected hash: 0 + (2 + 5·6)·4 = 128.
    cell_hash = Vector{Int32}(undef, 1)
    inv_cell_size = (1.0, 1.0, 1.0)
    num_cells = (Int32(4), Int32(6), Int32(8))
    Y = Float64[0.5; 2.5; 5.5;;]
    _assign_cells_kernel!(cell_hash, Y, inv_cell_size, num_cells)
    @test cell_hash[1] == Int32(128)
end

@testset "Position at the upper boundary is clamped to the last valid cell" begin
    # If a position reaches the kernel at Y_i = L_i (because fp roundoff or a
    # caller that bypassed `wrap_positions!`), Y_i · inv_cell_size_i lands at
    # m_i, and `floor(Int32, ·)` returns m_i — one past the last valid cell.
    # The min-clamp must bring it back to m_i - 1.
    cell_hash = Vector{Int32}(undef, 1)
    inv_cell_size = (1.0, 1.0, 1.0)
    num_cells = (Int32(4), Int32(4), Int32(4))
    Y = Float64[4.0; 4.0; 4.0;;]
    _assign_cells_kernel!(cell_hash, Y, inv_cell_size, num_cells)
    # Cell (3, 3, 3) → 3 + (3 + 3·4)·4 = 63.
    @test cell_hash[1] == Int32(63)
end

@testset "All hashes for a random position cloud fall below the total cell count" begin
    L = (4.0, 6.0, 8.0)
    inv_cell_size = (1.0, 1.0, 1.0)
    num_cells = (Int32(4), Int32(6), Int32(8))
    total = prod(Int(c) for c in num_cells)
    N = 1000
    Y = [L[i] * rand() for i in 1:3, _ in 1:N]
    cell_hash = Vector{Int32}(undef, N)
    _assign_cells_kernel!(cell_hash, Y, inv_cell_size, num_cells)
    @test all(0 .≤ cell_hash .< total)
end
