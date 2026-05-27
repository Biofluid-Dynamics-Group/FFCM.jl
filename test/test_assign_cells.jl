using Test
using FFCM
using FFCM: assign_cells!, _assign_cells_kernel!, wrap_positions!

@testset "assign_cells! agrees with the kernel called on the config fields" begin
    config = FFCMConfig{Float64}(; L = (4.0, 6.0, 8.0), R_c = 1.0, N = 64)
    Y = [config.L[i] * rand() for i in 1:3, _ in 1:length(config.cell_hash)]
    wrap_positions!(Y, config.L)
    Y_for_kernel = copy(Y)
    expected = Vector{Int32}(undef, length(config.cell_hash))
    _assign_cells_kernel!(expected, Y_for_kernel, config.inv_cell_size, config.num_cells)
    assign_cells!(config, Y)
    @test config.cell_hash == expected
end

@testset "assign_cells! returns the config's cell_hash buffer" begin
    config = FFCMConfig{Float64}(; L = (4.0, 4.0, 4.0), R_c = 1.0, N = 1)
    Y = Float64[0.5; 0.5; 0.5;;]
    returned = assign_cells!(config, Y)
    @test returned === config.cell_hash
end
