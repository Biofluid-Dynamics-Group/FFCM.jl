using Test

# Backend-selection checks that need no GPU. The construction and parity tests
# that require a functional device live in test_gpu_construction.jl, included
# under a `CUDA.functional()` guard.

@testset "GPU backend selection without a device" begin
    # `gpu_acceleration = true` must throw a clear `ArgumentError` when the CUDA
    # extension has not been loaded. The test session itself loads CUDA (for the
    # functional() guard), which provides the real assembly method and would mask
    # this path, so the check runs in a subprocess that loads ForceCouplingMethod but
    # not CUDA.
    @testset "errors clearly when the CUDA extension is not loaded" begin
        code = """
        using ForceCouplingMethod
        L = (4.0, 6.0, 8.0)
        try
            FFCMConfig(;
                L = L, R_c = 1.0, N = 8, kernel_widths_ratio = 2.0,
                num_grid_points = (Int32(8), Int32(12), Int32(16)), M_G = Int32(6),
                viscosity = 1.0, fft_planning = :estimate, gpu_acceleration = true,
            )
        catch err
            err isa ArgumentError && exit(42)
            rethrow()
        end
        exit(0)
        """
        cmd = `$(Base.julia_cmd()) --project=$(Base.active_project()) --startup-file=no -e $code`
        process = run(pipeline(ignorestatus(cmd); stdout = devnull, stderr = devnull))
        # exit 42 == threw ArgumentError; 0 == no throw; anything else == crash.
        @test process.exitcode == 42
    end
end
