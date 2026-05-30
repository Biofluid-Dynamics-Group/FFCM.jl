using Test
using BenchmarkTools
using FFCM
using FFCM: mobility!, FFCMMobility
using LinearAlgebra: mul!

# The hot path must allocate nothing: a single heap allocation per call would
# dominate the per-iteration cost in the downstream mobility solver. This covers
# the assembled driver and both mul! forms of the matrix-free operator.
@testset "Assembled mobility hot path does not allocate" begin
    for T in (Float32, Float64)
        N = 16
        config = FFCMConfig{T}(;
            L = (T(8), T(8), T(8)), R_c = T(1), N = N,
            Σ_over_σ = T(2), μ = T(1),
            num_grid_points = (Int32(16), Int32(16), Int32(16)), M_G = 8,
        )
        # Cluster particles so the pair-correction branch runs inside the driver.
        Y = Matrix{T}(undef, 3, N)
        F = Matrix{T}(undef, 3, N)
        for n in 1:N
            Y[1, n] = T(4) + T(0.25) * ((n - 1) % 4)
            Y[2, n] = T(4) + T(0.25) * ((n - 1) ÷ 4 % 4)
            Y[3, n] = T(4)
            F[1, n] = T(0.1) * n
            F[2, n] = T(-0.2) * n
            F[3, n] = T(0.3) * n
        end
        V = zeros(T, 3, N)
        M = FFCMMobility(config, Y)
        f = vec(F)
        v = zeros(T, 3N)

        @test (@ballocated mobility!($V, $config, $Y, $F)) == 0
        @test (@ballocated mul!($v, $M, $f)) == 0
        @test (@ballocated mul!($v, $M, $f, $(T(2)), $(T(0.5)))) == 0
    end
end
