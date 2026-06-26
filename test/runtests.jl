using Test
using CUDA

include("test_utilities.jl")

@testset "FFCM" begin
    include("accuracy/test_cell_geometry.jl")
    include("accuracy/test_wrap_positions.jl")
    include("accuracy/test_assign_cells_kernel.jl")
    include("accuracy/test_assign_cells.jl")
    include("api/test_assign_cells_api.jl")
    include("accuracy/test_sort_particles_by_cell.jl")
    include("api/test_sort_particles_by_cell_api.jl")
    include("accuracy/test_fcm_grid.jl")
    include("accuracy/test_modified_kernel_coefficients.jl")
    include("accuracy/test_spread_forces.jl")
    include("api/test_spread_forces_api.jl")
    include("accuracy/test_stokes_solve.jl")
    include("api/test_stokes_solve_api.jl")
    include("accuracy/test_interpolate_velocities.jl")
    include("api/test_interpolate_velocities_api.jl")
    include("accuracy/test_correct_velocities.jl")
    include("api/test_correct_velocities_api.jl")
    include("accuracy/test_mobility.jl")
    include("api/test_mobility_api.jl")
    include("accuracy/test_mobility_properties.jl")
    include("accuracy/test_single_sphere_mobility.jl")
    include("accuracy/test_fft_planning.jl")
    include("accuracy/test_fftw_threading.jl")
    include("hygiene/test_exported_surface.jl")
    include("hygiene/test_ascii_public_surface.jl")
    include("hygiene/test_aqua.jl")
    include("hygiene/test_jet.jl")

    # CUDA backend. The not-loaded error check needs no device; the construction
    # and parity tests run only where a CUDA device is functional.
    include("cuda/test_gpu_backend_selection.jl")
    if CUDA.functional()
        include("cuda/test_gpu_construction.jl")
    else
        @info "CUDA device not functional; skipping GPU construction tests"
    end
end
