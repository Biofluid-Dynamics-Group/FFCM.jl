using Test
using BenchmarkTools
using FFCM
using FFCM: mobility!, FFCMMobility
using LinearAlgebra: mul!

# Type stability and zero allocation are load-bearing for the hot path: an
# abstract return type would force dynamic dispatch, and a single heap
# allocation per call would dominate the per-iteration cost in the downstream
# mobility solver. This covers the assembled driver and both mul! forms of the
# matrix-free operator; `M * F` allocates its output by design, so it is
# checked for inference only. The clustered cloud puts many pairs within R_c
# so the pair-correction branch runs inside the driver.
@testset "Assembled-mobility hot-path guarantees" begin
    for T in (Float32, Float64)
        N = 16
        config = _standard_test_config(T; N = N)
        Y, F = _clustered_cloud(T, N)
        V = zeros(T, 3, N)
        M = FFCMMobility(config, Y)
        f = vec(F)
        v = zeros(T, 3N)
        @testset "type-stable ($T)" begin
            @inferred mobility!(V, config, Y, F)
            @inferred FFCMMobility(config, Y)
            @inferred mul!(v, M, f)
            @inferred mul!(v, M, f, T(2), T(0.5))
            @inferred M * F
        end
        @testset "allocation-free ($T)" begin
            @test (@ballocated mobility!($V, $config, $Y, $F) samples=1 evals=1) == 0
            @test (@ballocated mul!($v, $M, $f) samples=1 evals=1) == 0
            @test (@ballocated mul!($v, $M, $f, $(T(2)), $(T(0.5))) samples=1 evals=1) == 0
        end
    end
end
