using Test
using ForceCouplingMethod
using ForceCouplingMethod: _modified_kernel_coefficients

# The modified FCM kernel (Su & Keaveny 2024, §3 equation (22)) is
# (1 + (σ²−Σ²)/2 ⋅ Δ) Δ(x; Σ). Applying the Laplacian to the isotropic Gaussian
# collapses it to the polynomial (a₀ + a₂⋅r²)⋅Δ(x; Σ), evaluated through the
# separable 1-D weight inv_norm⋅exp(−x²⋅inv_2Σ²) per axis. Force spreading and
# velocity interpolation share these four scalars verbatim, because
# interpolation is the discrete adjoint of spreading.
@testset "Modified-kernel coefficients match the equation (22) expansion" begin
    for T in (Float32, Float64)
        σ = one(T) / sqrt(T(π))   # a = σ√π = 1, the unit-radius convention
        Σ = T(1.5) * σ

        a₀, a₂, inv_norm, inv_2Σ² = _modified_kernel_coefficients(σ, Σ)

        Σ² = Σ * Σ
        σ²_minus_Σ² = σ * σ - Σ²
        rtol = sqrt(eps(T))
        @test a₀ ≈ one(T) - T(3) * σ²_minus_Σ² / (T(2) * Σ²) rtol = rtol
        @test a₂ ≈ σ²_minus_Σ² / (T(2) * Σ² * Σ²) rtol = rtol
        @test inv_norm ≈ one(T) / sqrt(T(2) * T(π) * Σ²) rtol = rtol
        @test inv_2Σ² ≈ one(T) / (T(2) * Σ²) rtol = rtol
    end
end

@testset "Modified-kernel coefficients reduce to the plain Gaussian at Σ = σ" begin
    # Standard-FCM degenerate limit: the polynomial prefactor vanishes
    # (a₀ = 1, a₂ = 0) and the modified kernel is the unmodified Gaussian.
    for T in (Float32, Float64)
        σ = one(T) / sqrt(T(π))
        a₀, a₂, inv_norm, inv_2Σ² = _modified_kernel_coefficients(σ, σ)
        @test a₀ == one(T)
        @test a₂ == zero(T)
        @test inv_norm ≈ one(T) / sqrt(T(2) * T(π) * σ * σ) rtol = sqrt(eps(T))
        @test inv_2Σ² ≈ one(T) / (T(2) * σ * σ) rtol = sqrt(eps(T))
    end
end
