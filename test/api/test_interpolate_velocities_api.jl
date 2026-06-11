using Test
using BenchmarkTools
using FFCM
using FFCM: interpolate_velocities!

# Type stability and zero allocation are load-bearing for the hot path: an
# abstract return type would force dynamic dispatch, and a single heap
# allocation per call would dominate the per-iteration cost in the downstream
# mobility solver. The checks target the step entry point; its kernel is
# covered transitively (a zero-allocation entry bounds its callees, and JET
# walks inference into them — see test_jet.jl).
@testset "Velocity-interpolation hot-path guarantees" begin
    for T in (Float32, Float64)
        N = 32
        L = (T(8), T(8), T(8))
        config = _standard_test_config(
            T;
            N = N, L = L, num_grid_points = (Int32(8), Int32(8), Int32(8)),
        )
        _fill_sorted_midbox!(config, L, N)
        V = zeros(T, 3, N)
        @testset "type-stable ($T)" begin
            @inferred interpolate_velocities!(V, config)
        end
        @testset "allocation-free ($T)" begin
            @test (@ballocated interpolate_velocities!($V, $config) samples=1 evals=1) == 0
        end
    end
end
