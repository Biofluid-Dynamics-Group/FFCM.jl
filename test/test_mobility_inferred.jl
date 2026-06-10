using Test
using FFCM
using FFCM: mobility!, FFCMMobility
using LinearAlgebra: mul!

# Type stability is load-bearing for the hot path: an abstract return type would
# force dynamic dispatch and break the allocation-free guarantee.
@testset "Assembled mobility entry points are type-stable for Float32 and Float64" begin
    for T in (Float32, Float64)
        N = 4
        config = FFCMConfig{T}(;
            L = (T(8), T(8), T(8)), R_c = T(1), N = N,
            Σ_over_σ = T(2), μ = T(1),
            num_grid_points = (Int32(16), Int32(16), Int32(16)), M_G = 8,
        )
        Y = T[3.5 4.6 2.0 5.1; 4.0 4.2 6.0 4.0; 4.0 4.0 4.0 4.0]
        F = T[0.5 -0.3 0.2 0.1; -0.1 0.4 0.0 0.3; 0.2 0.1 0.7 -0.5]
        V = zeros(T, 3, N)
        @inferred mobility!(V, config, Y, F)

        M = @inferred FFCMMobility(config, Y)
        f = vec(F)
        v = zeros(T, 3N)
        @inferred mul!(v, M, f)
        @inferred mul!(v, M, f, T(2), T(0.5))
        @inferred M * F
    end
end
