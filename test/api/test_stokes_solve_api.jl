using Test
using BenchmarkTools
using FFCM
using FFCM: stokes_solve!

# Type stability and zero allocation are load-bearing for the hot path: an
# abstract return type would force dynamic dispatch, and a single heap
# allocation per call would dominate the per-iteration cost in the downstream
# mobility solver. The checks target the step entry point; its kernel is
# covered transitively (a zero-allocation entry bounds its callees, and JET
# walks inference into them — see test_jet.jl).
@testset "Stokes-solve hot-path guarantees" begin
    for T in (Float32, Float64)
        config = _standard_test_config(
            T;
            N = 1, L = (T(4), T(4), T(4)),
            num_grid_points = (Int32(8), Int32(8), Int32(8)),
        )
        @testset "type-stable ($T)" begin
            @inferred stokes_solve!(config)
        end
        @testset "allocation-free ($T)" begin
            @test (@ballocated stokes_solve!($config) samples=1 evals=1) == 0
        end
    end
end
