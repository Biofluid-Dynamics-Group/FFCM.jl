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
    # On a shared, queue-less node the GPU tests must run on exactly the one card the
    # user pinned with `CUDA_VISIBLE_DEVICES`; otherwise they would grab the default
    # device and could clobber another user's job. `CUDA.ndevices()` reflects that
    # env mask, so requiring a single visible device runs on a single-GPU box without
    # any env var, runs on a pinned card on the cluster, and skips (with the recipe)
    # when several GPUs are visible and none is pinned. See test/README.md.
    if !CUDA.functional()
        @info "CUDA not functional; skipping GPU tests"
    elseif CUDA.ndevices() != 1
        @info "$(CUDA.ndevices()) GPUs visible and none pinned; skipping GPU tests " *
              "to avoid grabbing a card in use on a shared node. Pick a free GPU " *
              "and pin it:\n" *
              "    gpustat            # or nvidia-smi\n" *
              "    export CUDA_VISIBLE_DEVICES=<index>\n" *
              "then re-run the tests."
    else
        gpu = CUDA.device()
        @info "Running GPU tests on device $(CUDA.deviceid(gpu)) ($(CUDA.name(gpu))); " *
              "free $(Base.format_bytes(CUDA.free_memory())) / " *
              "$(Base.format_bytes(CUDA.total_memory()))"
        include("cuda/test_gpu_construction.jl")
        include("cuda/test_gpu_cell_list.jl")
    end
end
