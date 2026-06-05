using Test
using FFCM
using FFCM: _self_correction, _correction_scalars,
    wrap_positions!, assign_cells!, sort_particles_by_cell!, correct_velocities!
using StaticArrays: SVector, SMatrix
using LinearAlgebra: I, norm, dot
using SpecialFunctions: erf

# Build a config whose grid parameters are irrelevant to the correction (it reads
# only the cell list and kernel widths). R_c = 1, L = 8 ⇒ 8 cells/axis, cell_size 1.
function _corr_config(::Type{T}; N, Σ_over_σ = T(2), μ = T(1), R_c = T(1)) where {T}
    L = (T(8), T(8), T(8))
    M = Int32(16)
    return FFCMConfig{T}(;
        L = L, R_c = R_c, N = N, Σ_over_σ = Σ_over_σ, μ = μ,
        num_grid_points = (M, M, M), M_G = 8,
    )
end

# Apply the step-6 correction operator to forces `F` at positions `Y`: runs the
# step-1/2 cell-list build, then `correct_velocities!` on a zeroed `V`. Linear in F.
function _apply_correction(config::FFCMConfig{T}, Y, F) where {T}
    Yc = copy(Y)
    wrap_positions!(Yc, config.L)
    assign_cells!(config, Yc)
    sort_particles_by_cell!(config, Yc, F)
    V = zeros(T, 3, size(Y, 2))
    correct_velocities!(V, config)
    return V
end

# Near-zero / absolute tolerance for quantities that should vanish.
_corr_atol(::Type{Float32}) = 1.0f-6
_corr_atol(::Type{Float64}) = 1.0e-10

# --- Independent oracle: the full FCM pairwise tensors (paper §2 equations
# (8)–(10) and (16)–(17), §3 equation (30)), assembled as 3×3 matrices. The
# correction is the difference of the two full mobilities,
# M^VF − M̃^VF = S(σ√2) − S(Σ√2) − (σ²−Σ²)Q(Σ√2) − ¼(σ²−Σ²)²T(Σ√2)
# (paper §3 equation (31)). This is an algebraically-distinct route to the same
# tensor the implementation collapses into A·I + B·xxᵀ, so it cross-checks the
# collapse and the equation-(31) erf-argument. `s` is the √2-scaled width.

_gaussian(r², s², ::Type{T}) where {T} =
    (T(2) * T(π) * s²)^(-T(3) / 2) * exp(-r² / (T(2) * s²))

function _S_tensor(x::SVector{3,T}, s::T, μ::T) where {T}
    r = norm(x); r² = r * r; s² = s * s
    e = erf(r / (s * sqrt(T(2))))
    Δ = _gaussian(r², s², T)
    xx = x * x'
    cI = e / (T(8) * T(π) * μ * r) * (one(T) + s² / r²) - s²^2 / (T(2) * μ * r²) * Δ
    cxx = e / (T(8) * T(π) * μ * r^3) * (one(T) - T(3) * s² / r²) +
          T(3) * s²^2 / (T(2) * μ * r^4) * Δ
    return cI * SMatrix{3,3,T}(I) + cxx * xx
end

function _Q_tensor(x::SVector{3,T}, s::T, μ::T) where {T}
    r = norm(x); r² = r * r; s² = s * s
    e = erf(r / (s * sqrt(T(2))))
    Δ = _gaussian(r², s², T)
    xx = x * x'
    cI = e / (T(4) * T(π) * μ * r^3) - (one(T) + s² / r²) * Δ / μ
    cxx = -T(3) * e / (T(4) * T(π) * μ * r^5) + (one(T) + T(3) * s² / r²) / (μ * r²) * Δ
    return cI * SMatrix{3,3,T}(I) + cxx * xx
end

function _T_tensor(x::SVector{3,T}, s::T, μ::T) where {T}
    r = norm(x); r² = r * r; s² = s * s
    Δ = _gaussian(r², s², T)
    xx = x * x'
    cI = (T(2) - r² / s²) / (μ * s²) * Δ
    cxx = one(T) / (μ * s²^2) * Δ
    return cI * SMatrix{3,3,T}(I) + cxx * xx
end

function _correction_tensor_oracle(x::SVector{3,T}, σ::T, Σ::T, μ::T) where {T}
    σ²_minus_Σ² = σ^2 - Σ^2
    sσ = σ * sqrt(T(2))
    sΣ = Σ * sqrt(T(2))
    return _S_tensor(x, sσ, μ) - _S_tensor(x, sΣ, μ) -
           σ²_minus_Σ² * _Q_tensor(x, sΣ, μ) - σ²_minus_Σ²^2 / 4 * _T_tensor(x, sΣ, μ)
end

@testset "Standard-FCM degenerate limit (Σ = σ) gives zero correction" begin
    # At Σ = σ the grid kernel is the true kernel, so M̃ = M and the correction
    # vanishes identically — both the self term and every pair scalar (paper §4,
    # the Σ = σ degenerate limit).
    for T in (Float32, Float64)
        a = one(T)
        σ = a / sqrt(T(π))
        for μ in (T(0.5), T(1), T(2.7))
            @test isapprox(_self_correction(σ, σ, a, μ), zero(T); atol = _corr_atol(T))
            for r in (T(0.2), T(0.75), T(1.5))
                A, B = _correction_scalars(r, σ, σ, μ)
                @test isapprox(A, zero(T); atol = _corr_atol(T))
                @test isapprox(B, zero(T); atol = _corr_atol(T))
            end
        end
    end
end

@testset "Correction scalars reproduce the full-tensor difference of mobilities" begin
    # `_correction_scalars` returns (A, B) with the correction tensor A·I + B·xxᵀ
    # (un-normalised xxᵀ). Cross-check against the independent full-tensor oracle
    # over a range of separations, directions, and parameters.
    for T in (Float32, Float64)
        a = one(T)
        σ = a / sqrt(T(π))
        directions = (
            SVector{3,T}(1, 0, 0),
            SVector{3,T}(0, 1, 0),
            SVector{3,T}(1, 1, 1) / sqrt(T(3)),
            SVector{3,T}(2, -1, 0.5) / norm(SVector{3,T}(2, -1, 0.5)),
        )
        for Σ_over_σ in (T(1.5), T(2.5)), μ in (T(0.7), T(1.6))
            Σ = Σ_over_σ * σ
            for r in (T(0.3), T(0.6), T(1.2)), d in directions
                x = r * d
                A, B = _correction_scalars(r, σ, Σ, μ)
                M_impl = A * SMatrix{3,3,T}(I) + B * (x * x')
                M_oracle = _correction_tensor_oracle(x, σ, Σ, μ)
                @test M_impl ≈ M_oracle rtol = sqrt(eps(T)) atol = _corr_atol(T)
            end
        end
    end
end

@testset "Self correction matches the r → 0 limit (paper Appendix B eq (B.1))" begin
    # Paper Appendix B, equation (B.1): the well-defined r → 0 diagonal of the
    # pairwise correction, a scalar × I added to every particle. Pin
    # `_self_correction` against the closed form written in an independent
    # factorisation, over a sweep of Σ/σ and μ. a = σ√π (paper §2).
    for T in (Float32, Float64)
        a = one(T)
        σ = a / sqrt(T(π))
        for Σ_over_σ in (T(1.25), T(2), T(3.5)), μ in (T(0.5), T(1), T(2.7))
            Σ = Σ_over_σ * σ
            Σπ = Σ * sqrt(T(π))
            σ²_minus_Σ² = σ^2 - Σ^2
            # Reference: paper Appendix B equation (B.1), grouped term by term.
            stokes = one(T) / (T(6) * T(π) * μ * a)
            mod_stokes = one(T) / (T(6) * T(π) * μ * Σπ)
            pd_term = σ²_minus_Σ² / (T(12) * μ * Σπ^3)
            bilap = σ²_minus_Σ²^2 / (T(32) * μ * Σ^5 * T(π)^(T(3) / 2))
            reference = stokes - mod_stokes + pd_term - bilap

            self_correction_term = _self_correction(σ, Σ, a, μ)
            @test self_correction_term ≈ reference rtol = sqrt(eps(T))
        end
    end
end

@testset "Isolated particle gets only the self term" begin
    # A particle with no neighbour within R_c receives only the diagonal self
    # correction: V[:, n] += self_correction_term · F_n.
    for T in (Float32, Float64)
        config = _corr_config(T; N = 2)
        a, σ, Σ, μ = config.a, config.σ, config.Σ, config.μ
        c = _self_correction(σ, Σ, a, μ)
        # Two particles separated well beyond R_c = 1 (distance √27 ≈ 5.2).
        Y = T[1 4; 1 4; 1 4]
        F = T[0.3 -0.5; -0.7 0.9; 1.1 0.2]
        V = _apply_correction(config, Y, F)
        @test V[:, 1] ≈ c .* F[:, 1] rtol = sqrt(eps(T)) atol = _corr_atol(T)
        @test V[:, 2] ≈ c .* F[:, 2] rtol = sqrt(eps(T)) atol = _corr_atol(T)
    end
end

@testset "Two-particle pair correction matches the difference-of-mobilities tensor" begin
    # Velocity of each particle = self term + (correction tensor)·(neighbour force).
    # The tensor is symmetric in x = Y_n − Y_m, so both partners share it.
    for T in (Float32, Float64)
        config = _corr_config(T; N = 2)
        a, σ, Σ, μ = config.a, config.σ, config.Σ, config.μ
        c = _self_correction(σ, Σ, a, μ)
        Y = T[4 4.5; 4 4.2; 4 3.8]   # particle 1 = (4,4,4), particle 2 = (4.5,4.2,3.8)
        F = T[0.3 -0.5; -0.7 0.9; 1.1 0.2]
        V = _apply_correction(config, Y, F)
        x = SVector{3,T}(Y[1, 1] - Y[1, 2], Y[2, 1] - Y[2, 2], Y[3, 1] - Y[3, 2])
        @test norm(x) < config.R_c
        M = _correction_tensor_oracle(x, σ, Σ, μ)
        F1 = SVector{3,T}(F[1, 1], F[2, 1], F[3, 1])
        F2 = SVector{3,T}(F[1, 2], F[2, 2], F[3, 2])
        @test V[:, 1] ≈ c .* F[:, 1] + M * F2 rtol = sqrt(eps(T)) atol = _corr_atol(T)
        @test V[:, 2] ≈ c .* F[:, 2] + M * F1 rtol = sqrt(eps(T)) atol = _corr_atol(T)
    end
end

@testset "Pair correction is symmetric (SPD precondition)" begin
    # Fₐᵀ (corr Fᵦ) = Fᵦᵀ (corr Fₐ): the correction operator is self-adjoint, so the
    # split mobility stays symmetric positive-definite (paper §3).
    for T in (Float32, Float64)
        config = _corr_config(T; N = 5)
        Y = T[4.0 4.4 3.7 4.2 3.9;
              4.0 3.8 4.3 4.1 3.6;
              4.0 4.2 4.0 3.7 4.4]
        Fa = T[0.3 -0.5 0.8 -0.2 0.6;
               -0.7 0.9 -0.1 0.5 -0.4;
               1.1 0.2 -0.6 0.3 0.7]
        Fb = T[-0.2 0.6 -0.9 0.4 0.1;
               0.5 -0.3 0.7 -0.8 0.2;
               -0.4 0.1 0.3 0.6 -0.5]
        Va = _apply_correction(config, Y, Fb)
        Vb = _apply_correction(config, Y, Fa)
        @test dot(Fa, Va) ≈ dot(Fb, Vb) rtol = sqrt(eps(T))
    end
end

@testset "Correction is linear in F and invariant under translation/periodicity" begin
    for T in (Float32, Float64)
        config = _corr_config(T; N = 4)
        Y = T[4.0 4.4 3.7 4.2;
              4.0 3.8 4.3 4.1;
              4.0 4.2 4.0 3.7]
        F1 = T[0.3 -0.5 0.8 -0.2;
               -0.7 0.9 -0.1 0.5;
               1.1 0.2 -0.6 0.3]
        F2 = T[-0.2 0.6 -0.9 0.4;
               0.5 -0.3 0.7 -0.8;
               -0.4 0.1 0.3 0.6]
        α, β = T(1.7), T(-0.8)
        V_combined = _apply_correction(config, Y, α .* F1 .+ β .* F2)
        V_separate = α .* _apply_correction(config, Y, F1) .+
                     β .* _apply_correction(config, Y, F2)
        @test V_combined ≈ V_separate rtol = sqrt(eps(T)) atol = _corr_atol(T)

        # Rigid translation of every particle leaves relative separations (hence the
        # correction) unchanged, even though cell assignment changes.
        V_ref = _apply_correction(config, Y, F1)
        shift = T[1.3, -2.1, 0.7]
        V_shift = _apply_correction(config, Y .+ shift, F1)
        @test V_shift ≈ V_ref rtol = sqrt(eps(T)) atol = _corr_atol(T)

        # Periodic image: move one particle by L along x ⇒ identical after wrapping.
        Y_img = copy(Y)
        Y_img[1, 2] += config.L[1]
        V_img = _apply_correction(config, Y_img, F1)
        @test V_img ≈ V_ref rtol = sqrt(eps(T)) atol = _corr_atol(T)
    end
end

@testset "Correction adds to V rather than overwriting it" begin
    for T in (Float32, Float64)
        config = _corr_config(T; N = 2)
        Y = T[4 4.5; 4 4.2; 4 3.8]
        F = T[0.3 -0.5; -0.7 0.9; 1.1 0.2]
        V_only = _apply_correction(config, Y, F)
        # Re-run, but start from a pre-filled V (as interpolate_velocities! would leave).
        Yc = copy(Y)
        wrap_positions!(Yc, config.L)
        assign_cells!(config, Yc)
        sort_particles_by_cell!(config, Yc, F)
        V0 = T[2.0 -1.0; 0.5 3.0; -2.5 1.5]
        V = copy(V0)
        correct_velocities!(V, config)
        @test V ≈ V0 .+ V_only rtol = sqrt(eps(T)) atol = _corr_atol(T)
    end
end
