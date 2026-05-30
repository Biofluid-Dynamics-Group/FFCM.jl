using Test
using FFCM
using FFCM: mobility!, FFCMMobility
using LinearAlgebra: mul!, dot

# Tolerance convention (spec/mobility.md, CLAUDE.md §2): relative comparisons use
# rtol = sqrt(eps(T)); the atol floor catches components that come out ≈ 0, where
# a relative tolerance is ill-defined.
_mobility_atol(::Type{Float32}) = 1.0f-6
_mobility_atol(::Type{Float64}) = 1.0e-10

# A small multi-particle configuration for the assembled-operator tests. Grid and
# kernel widths follow the other step tests (Δx isotropic, M_G < M).
function _mobility_test_config(::Type{T}, N) where {T}
    return FFCMConfig{T}(;
        L = (T(8), T(8), T(8)), R_c = T(1), N = N,
        Σ_over_σ = T(2), μ = T(1),
        num_grid_points = (Int32(16), Int32(16), Int32(16)), M_G = 8,
    )
end

@testset "mobility! leaves the caller's positions and forces unmodified" begin
    # The hot call folds positions into [0, L) internally (into a config-owned
    # buffer); the contract is that the caller's Y and F arrays are read-only
    # (spec/mobility.md). Include out-of-domain coordinates so wrapping has work.
    for T in (Float32, Float64)
        N = 4
        config = _mobility_test_config(T, N)
        Y = T[9.3 4.1 -0.4 4.0; 4.0 4.2 4.0 9.9; 4.0 4.0 4.0 4.0]
        F = T[0.5 -0.3 0.2 0.1; -0.1 0.4 0.0 0.3; 0.2 0.1 0.7 -0.5]
        Y0 = copy(Y)
        F0 = copy(F)
        V = zeros(T, 3, N)
        mobility!(V, config, Y, F)
        @test Y == Y0
        @test F == F0
    end
end

@testset "mobility! is linear in the forces" begin
    # M^VF is a linear operator: M(aF + bG) = a·MF + b·MG.
    for T in (Float32, Float64)
        N = 3
        config = _mobility_test_config(T, N)
        Y = T[3.5 4.6 2.0; 4.0 4.2 6.0; 4.0 4.0 4.0]
        F = T[0.5 -0.3 0.2; -0.1 0.4 0.0; 0.2 0.1 0.7]
        G = T[0.1 0.2 -0.4; 0.3 -0.5 0.6; -0.2 0.4 0.1]
        a = T(1.7)
        b = T(-0.9)
        V_F = zeros(T, 3, N)
        V_G = zeros(T, 3, N)
        V_c = zeros(T, 3, N)
        mobility!(V_F, config, Y, F)
        mobility!(V_G, config, Y, G)
        mobility!(V_c, config, Y, a .* F .+ b .* G)
        @test isapprox(
            V_c, a .* V_F .+ b .* V_G;
            rtol = sqrt(eps(T)), atol = _mobility_atol(T),
        )
    end
end

@testset "Assembled mobility is symmetric positive-definite" begin
    # M^VF = J·L⁻¹·J† + (M − M̃) is SPD: interpolation is the discrete adjoint of
    # spreading, the Stokes solve is self-adjoint, and the correction is a
    # symmetric pair tensor (spec/mobility.md). Two particles 0.7 apart fall
    # within R_c = 1, so the off-diagonal pair coupling is exercised.
    for T in (Float32, Float64)
        N = 2
        config = _mobility_test_config(T, N)
        Y = T[3.8 4.5; 4.0 4.0; 4.0 4.0]
        F = T[0.5 -0.3; -0.1 0.4; 0.2 0.6]
        G = T[0.1 0.7; 0.3 -0.2; -0.4 0.5]
        V_F = zeros(T, 3, N)
        V_G = zeros(T, 3, N)
        mobility!(V_F, config, Y, F)
        mobility!(V_G, config, Y, G)
        @test dot(G, V_F) ≈ dot(F, V_G) rtol = sqrt(eps(T))
        @test dot(F, V_F) > zero(T)
    end
end

@testset "FFCMMobility reports operator dimensions" begin
    for T in (Float32, Float64)
        N = 3
        config = _mobility_test_config(T, N)
        Y = T[3.5 4.6 2.0; 4.0 4.2 6.0; 4.0 4.0 4.0]
        M = FFCMMobility(config, Y)
        @test size(M) == (3N, 3N)
        @test size(M, 1) == 3N
        @test size(M, 2) == 3N
        @test eltype(M) == T
    end
end

@testset "Three-argument mul! matches mobility!" begin
    # The matrix-free operator is the flat-vector view of mobility!: vec(MF)
    # column-major (spec/mobility.md).
    for T in (Float32, Float64)
        N = 3
        config = _mobility_test_config(T, N)
        Y = T[3.5 4.6 2.0; 4.0 4.2 6.0; 4.0 4.0 4.0]
        F = T[0.5 -0.3 0.2; -0.1 0.4 0.0; 0.2 0.1 0.7]
        V_mat = zeros(T, 3, N)
        mobility!(V_mat, config, Y, F)
        M = FFCMMobility(config, Y)
        v = zeros(T, 3N)
        mul!(v, M, vec(F))
        @test isapprox(
            v, vec(V_mat); rtol = sqrt(eps(T)), atol = _mobility_atol(T),
        )
    end
end

@testset "Five-argument mul! computes α·M·f + β·v" begin
    for T in (Float32, Float64)
        N = 3
        config = _mobility_test_config(T, N)
        Y = T[3.5 4.6 2.0; 4.0 4.2 6.0; 4.0 4.0 4.0]
        F = T[0.5 -0.3 0.2; -0.1 0.4 0.0; 0.2 0.1 0.7]
        M = FFCMMobility(config, Y)
        f = vec(F)
        ref = zeros(T, 3N)
        mul!(ref, M, f)                       # ref = M·f
        α = T(2)
        β = T(-0.5)
        v0 = T[T(0.1) * k for k in 1:(3N)]
        v = copy(v0)
        mul!(v, M, f, α, β)
        @test isapprox(
            v, α .* ref .+ β .* v0;
            rtol = sqrt(eps(T)), atol = _mobility_atol(T),
        )
        # β = 0 overwrites v, ignoring its prior (here non-finite) contents.
        v_nan = fill(T(NaN), 3N)
        mul!(v_nan, M, f, α, zero(T))
        @test all(isfinite, v_nan)
        @test isapprox(
            v_nan, α .* ref; rtol = sqrt(eps(T)), atol = _mobility_atol(T),
        )
    end
end

@testset "FFCMMobility is symmetric positive-definite through mul!" begin
    for T in (Float32, Float64)
        N = 2
        config = _mobility_test_config(T, N)
        Y = T[3.8 4.5; 4.0 4.0; 4.0 4.0]
        M = FFCMMobility(config, Y)
        f = T[0.5, -0.1, 0.2, -0.3, 0.4, 0.6]
        g = T[0.1, 0.3, -0.4, 0.7, -0.2, 0.5]
        Mf = zeros(T, 6)
        Mg = zeros(T, 6)
        mul!(Mf, M, f)
        mul!(Mg, M, g)
        @test dot(g, Mf) ≈ dot(f, Mg) rtol = sqrt(eps(T))
        @test dot(f, Mf) > zero(T)
    end
end
