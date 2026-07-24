using Test
using BenchmarkTools
using ForceCouplingMethod
using ForceCouplingMethod: assign_cells!, wrap_positions!

# Type stability and zero allocation are load-bearing for the hot path: an
# abstract return type would force dynamic dispatch, and a single heap
# allocation per call would dominate the per-iteration cost in the downstream
# mobility solver. The checks target the step entry points; their kernels are
# covered transitively (a zero-allocation entry bounds its callees, and JET
# walks inference into them — see test_jet.jl).
@testset "Position wrapping and cell assignment hot-path guarantees" begin
    for T in (Float32, Float64)
        N = 256
        L = (T(4), T(6), T(8))
        config = FFCMConfig{T}(;
            L = L, R_c = T(1), N = N, _fcm_grid_kwargs(L)...,
        )
        Y = T[
            L[i] * (T(0.05) + T(0.9) * T(n - 1) / T(N - 1)) for i in 1:3, n in 1:N
        ]
        @testset "type-stable ($T)" begin
            @inferred wrap_positions!(Y, config.L)
            @inferred assign_cells!(config, Y)
        end
        @testset "allocation-free ($T)" begin
            @test (@ballocated wrap_positions!($Y, $(config.L)) samples=1 evals=1) == 0
            @test (@ballocated assign_cells!($config, $Y) samples=1 evals=1) == 0
        end
    end
end
