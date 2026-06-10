using Test

# Test helper: picks `Σ_over_σ`, `num_grid_points`, and `M_G` for tests that
# only exercise the step-1 / step-2 cell-list machinery and do not care
# about the FCM grid. `num_grid_points` is sized so h is isotropic across
# axes (paper §3 assumption); a target h of `L[1] / 8` works for every L
# the existing tests use.
function _fcm_grid_kwargs(L::NTuple{3, T}) where {T}
    h_target = L[1] / 8
    num_grid_points = ntuple(i -> Int32(round(Int, L[i] / h_target)), 3)
    return (
        Σ_over_σ = T(2),
        num_grid_points = num_grid_points,
        # The stencil cannot exceed the grid on any axis; clamp for thin domains
        # where a short axis yields fewer than 8 grid points.
        M_G = min(8, Int(minimum(num_grid_points))),
        μ = T(1),
    )
end

@testset "FFCM" begin
    include("test_cell_geometry.jl")
    include("test_wrap_positions.jl")
    include("test_assign_cells_kernel.jl")
    include("test_assign_cells.jl")
    include("test_assign_cells_inferred.jl")
    include("test_assign_cells_allocations.jl")
    include("test_sort_particles_by_cell.jl")
    include("test_sort_particles_by_cell_inferred.jl")
    include("test_sort_particles_by_cell_allocations.jl")
    include("test_fcm_grid.jl")
    include("test_modified_kernel_coefficients.jl")
    include("test_spread_forces.jl")
    include("test_spread_forces_inferred.jl")
    include("test_spread_forces_allocations.jl")
    include("test_stokes_solve.jl")
    include("test_stokes_solve_inferred.jl")
    include("test_stokes_solve_allocations.jl")
    include("test_interpolate_velocities.jl")
    include("test_interpolate_velocities_inferred.jl")
    include("test_interpolate_velocities_allocations.jl")
    include("test_correct_velocities.jl")
    include("test_correct_velocities_inferred.jl")
    include("test_correct_velocities_allocations.jl")
    include("test_mobility.jl")
    include("test_mobility_inferred.jl")
    include("test_mobility_allocations.jl")
    include("test_single_sphere_mobility.jl")
    include("test_exported_surface.jl")
    include("test_aqua.jl")
    include("test_jet.jl")
end
