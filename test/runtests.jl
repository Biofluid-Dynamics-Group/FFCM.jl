using Test

@testset "FFCM" begin
    include("test_cell_geometry.jl")
    include("test_wrap_positions.jl")
    include("test_assign_cells_kernel.jl")
    include("test_assign_cells.jl")
    include("test_assign_cells_inferred.jl")
    include("test_assign_cells_allocations.jl")
    include("test_aqua.jl")
    include("test_jet.jl")
end
