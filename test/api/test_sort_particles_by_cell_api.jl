using Test
using BenchmarkTools
using FFCM
using FFCM: sort_particles_by_cell!, assign_cells!, wrap_positions!

# Type stability and zero allocation are load-bearing for the hot path: an
# abstract return type would force dynamic dispatch, and a single heap
# allocation per call would dominate the per-iteration cost in the downstream
# mobility solver. The checks target the step entry point; its kernels are
# covered transitively (a zero-allocation entry bounds its callees, and JET
# walks inference into them — see test_jet.jl).
@testset "Cell-sort hot-path guarantees" begin
    for T in (Float32, Float64)
        N = 256
        L = (T(4), T(6), T(8))
        config = FFCMConfig{T}(;
            L = L, R_c = T(1), N = N, _fcm_grid_kwargs(L)...,
        )
        Y = T[
            L[i] * (T(0.05) + T(0.9) * T(n - 1) / T(N - 1)) for i in 1:3, n in 1:N
        ]
        F = T[T(0.1) * n * (i - 2) for i in 1:3, n in 1:N]
        wrap_positions!(Y, config.L)
        assign_cells!(config, Y)
        @testset "type-stable ($T)" begin
            @inferred sort_particles_by_cell!(config, Y, F)
        end
        @testset "allocation-free ($T)" begin
            allocations =
                @ballocated sort_particles_by_cell!($config, $Y, $F) samples=1 evals=1
            @test allocations == 0
        end
    end
end
