using Test
using FFCM
using FFCM: spread_forces!, _spread_forces_kernel!
using StaticArrays
using StructArrays: components

@testset "Single-particle spread matches the closed-form modified kernel" begin
    # Single particle at the box centre; unit force along x; cubic grid
    # large enough that the M_G^3 stencil covers every grid point so we can
    # check every entry of force_grid against the closed form.
    for T in (Float32, Float64)
        L = (T(4), T(4), T(4))
        M = Int32(8)
        config = FFCMConfig{T}(;
            L = L, R_c = T(1), N = 1,
            Σ_over_σ = T(2),
            num_grid_points = (M, M, M),
            M_G = 8,
        )
        config.Y_sorted[1, 1] = T(2)
        config.Y_sorted[2, 1] = T(2)
        config.Y_sorted[3, 1] = T(2)
        config.F_sorted[1, 1] = T(1)
        config.F_sorted[2, 1] = T(0)
        config.F_sorted[3, 1] = T(0)
        spread_forces!(config)

        # Reference computed fresh from paper eq 267 expanded as
        # Δ + (σ² - Σ²)/2 · ∇²Δ, where ∇²Δ = (r²/Σ⁴ - 3/Σ²) Δ.
        σ, Σ, Δx = config.σ, config.Σ, config.Δx
        Y_n = (T(2), T(2), T(2))
        F_n = (T(1), T(0), T(0))
        rtol = sqrt(eps(T))
        @testset "grid point ($ix, $iy, $iz)" for iz in 1:M, iy in 1:M, ix in 1:M
            x_g = ((ix - 1) * Δx, (iy - 1) * Δx, (iz - 1) * Δx)
            r² = (x_g[1] - Y_n[1])^2 + (x_g[2] - Y_n[2])^2 + (x_g[3] - Y_n[3])^2
            Δ = (T(2) * T(π) * Σ^2)^(-T(3)/T(2)) * exp(-r²/(T(2) * Σ^2))
            laplacian = (r²/Σ^4 - T(3)/Σ^2) * Δ
            Δ̃ = Δ + (σ^2 - Σ^2)/T(2) * laplacian
            expected = SVector{3, T}(F_n[1] * Δ̃, F_n[2] * Δ̃, F_n[3] * Δ̃)
            @test config.force_grid[ix, iy, iz] ≈ expected rtol=rtol
        end
    end
end

@testset "Stencil anchors on the nearest grid point, biased low for even M_G (cuFCM convention)" begin
    # For Y/Δx = 0.3 the nearest grid point is index 0 (0-based); for
    # Y/Δx = 0.7 it is index 1. With M_G even the stencil extends M_G/2
    # below the anchor and M_G/2 - 1 above (paper §5 Table 2 calibration
    # convention, matching cuFCM `my_rint(Y/dx) - ngd/2 + (i mod ngd)`).
    # We pick `M_G = 4` and an `M_x = 16` grid so the stencil is strictly
    # smaller than the grid and the lower- and upper-half anchors give
    # distinct, periodic-wrapped support sets.
    for T in (Float32, Float64)
        L = (T(8), T(8), T(8))
        M = Int32(16)
        config = FFCMConfig{T}(;
            L = L, R_c = T(1), N = 1,
            Σ_over_σ = T(2),
            num_grid_points = (M, M, M),
            M_G = 4,
        )
        Δx = config.Δx

        spread_at = function (Y_frac)
            fill!(config.Y_sorted, T(Y_frac) * Δx)
            config.F_sorted[1, 1] = T(1)
            config.F_sorted[2, 1] = T(0)
            config.F_sorted[3, 1] = T(0)
            spread_forces!(config)
            fx = components(config.force_grid)[1]
            marginal_x = vec(sum(abs.(fx); dims = (2, 3)))
            return sort(findall(>(zero(T)), marginal_x))
        end

        # Y/Δx = 0.3 → anchor = 0 → 0-based stencil {-2, -1, 0, 1}
        # → wrapped mod 16: {14, 15, 0, 1} → 1-based: {15, 16, 1, 2}.
        @test spread_at(T(0.3)) == sort([15, 16, 1, 2])

        # Y/Δx = 0.7 → anchor = 1 → 0-based stencil {-1, 0, 1, 2}
        # → wrapped mod 16: {15, 0, 1, 2} → 1-based: {16, 1, 2, 3}.
        @test spread_at(T(0.7)) == sort([16, 1, 2, 3])
    end
end

@testset "Total spread force equals total particle force within truncation tolerance" begin
    # ∫ Δ̃(x; Σ) d³x = 1 exactly (the Laplacian of a Gaussian integrates
    # to zero), so Σₙ Fₙ should be conserved by the spread up to the
    # truncation error of the M_G^3 stencil and the Riemann-sum
    # discretisation of the integral. Paper Table 2 (`outline.tex:600`)
    # tabulates this error for chosen (M_G, Σ/Δx).
    for T in (Float32, Float64)
        # Parameters chosen so the stencil radius `(M_G/2)·Δx ≈ 7.1·Σ`
        # captures the Gaussian to ~1 - erf(7.1/√2)³ ≈ 0 mass — far above
        # the ~3.4·Σ threshold for sub-percent per-particle conservation.
        # `Σ/Δx ≈ 1.13` matches the paper's lowest-tolerance regime
        # (paper Table 2, `outline.tex:600`). The box `L = 16` keeps the
        # nearest periodic image at ~7·Σ from any particle, where the
        # kernel value is < 1e-10 — far below the rounding floor.
        L = (T(16), T(16), T(16))
        M = Int32(16)
        N = 27   # a 3×3×3 sub-grid
        config = FFCMConfig{T}(;
            L = L, R_c = T(1), N = N,
            Σ_over_σ = T(2),
            num_grid_points = (M, M, M),
            M_G = 16,
        )
        # Particles on a 3×3×3 lattice with phase-coherent forces. The
        # specific positions and forces do not matter for the invariant;
        # we just need to break the trivial single-particle case.
        for n in 1:N
            nx = ((n - 1) % 3) + 1
            ny = ((n - 1) ÷ 3) % 3 + 1
            nz = ((n - 1) ÷ 9) + 1
            config.Y_sorted[1, n] = L[1] * (nx - T(0.5)) / 3
            config.Y_sorted[2, n] = L[2] * (ny - T(0.5)) / 3
            config.Y_sorted[3, n] = L[3] * (nz - T(0.5)) / 3
            config.F_sorted[1, n] = sin(T(n))
            config.F_sorted[2, n] = cos(T(n))
            config.F_sorted[3, n] = T(n) / N - T(0.5)
        end
        spread_forces!(config)

        fx, fy, fz = components(config.force_grid)
        Δx³ = config.Δx^3
        total_grid = (sum(fx) * Δx³, sum(fy) * Δx³, sum(fz) * Δx³)
        total_force = ntuple(i -> sum(@view config.F_sorted[i, :]), 3)

        # Tolerance follows the paper-tabulated error for the chosen
        # (M_G, Σ/Δx). At Σ/Δx ≈ 9 with M_G = 12 the truncation error is
        # well below `1e-6` in single precision and well below `1e-8` in
        # double; we use `sqrt(eps(T))` as a single relaxed bound.
        rtol = sqrt(eps(T))
        @test total_grid[1] ≈ total_force[1] rtol=rtol
        @test total_grid[2] ≈ total_force[2] rtol=rtol
        @test total_grid[3] ≈ total_force[3] rtol=rtol
    end
end

@testset "First moment of the spread is the particle position (within truncation tolerance)" begin
    # ∫ x · Δ̃ₙ(x; Σ) d³x = Yₙ (Laplacian of Gaussian has zero first
    # moment, so the modified kernel has the same centroid as the plain
    # Gaussian).
    for T in (Float32, Float64)
        L = (T(16), T(16), T(16))
        M = Int32(16)
        config = FFCMConfig{T}(;
            L = L, R_c = T(1), N = 1,
            Σ_over_σ = T(2),
            num_grid_points = (M, M, M),
            M_G = 16,
        )
        Y_n = (T(8.3), T(7.7), T(8.1))   # off-grid, near the box centre
                                          # so the M_G stencil never wraps
        config.Y_sorted[1, 1] = Y_n[1]
        config.Y_sorted[2, 1] = Y_n[2]
        config.Y_sorted[3, 1] = Y_n[3]
        config.F_sorted[1, 1] = T(1)
        config.F_sorted[2, 1] = T(0)
        config.F_sorted[3, 1] = T(0)
        spread_forces!(config)

        fx, _, _ = components(config.force_grid)
        Δx = config.Δx
        Δx³ = Δx^3
        centroid = (zero(T), zero(T), zero(T))
        @inbounds for iz in 1:M, iy in 1:M, ix in 1:M
            x_g = ((ix - 1) * Δx, (iy - 1) * Δx, (iz - 1) * Δx)
            w = fx[ix, iy, iz] * Δx³
            centroid = (
                centroid[1] + x_g[1] * w,
                centroid[2] + x_g[2] * w,
                centroid[3] + x_g[3] * w,
            )
        end
        rtol = sqrt(eps(T))
        @test centroid[1] ≈ Y_n[1] rtol=rtol
        @test centroid[2] ≈ Y_n[2] rtol=rtol
        @test centroid[3] ≈ Y_n[3] rtol=rtol
    end
end

@testset "A particle translation by Δx shifts the spread by one grid cell" begin
    # The stencil indexing is periodic on the canonical grid; translating
    # the particle by Δx along axis i shifts every stencil grid point by
    # one cell on axis i (mod M_i). The kernel weights are unchanged.
    for T in (Float32, Float64)
        L = (T(8), T(8), T(8))
        M = Int32(16)
        config = FFCMConfig{T}(;
            L = L, R_c = T(1), N = 1,
            Σ_over_σ = T(2),
            num_grid_points = (M, M, M),
            M_G = 8,
        )
        spread_at = function (Y)
            config.Y_sorted[1, 1] = Y[1]
            config.Y_sorted[2, 1] = Y[2]
            config.Y_sorted[3, 1] = Y[3]
            config.F_sorted[1, 1] = T(1)
            config.F_sorted[2, 1] = T(2)
            config.F_sorted[3, 1] = -T(3)
            spread_forces!(config)
            fx, fy, fz = components(config.force_grid)
            return (copy(fx), copy(fy), copy(fz))
        end
        Δx = config.Δx
        Y_a = (T(2.4), T(3.7), T(5.2))
        fa = spread_at(Y_a)
        fc = spread_at((Y_a[1] + Δx, Y_a[2], Y_a[3]))
        rtol = sqrt(eps(T))
        @test all(isapprox(fa[1][ix, iy, iz], fc[1][mod1(ix + 1, Int(M)), iy, iz]; rtol = rtol)
                  for iz in 1:M, iy in 1:M, ix in 1:M)
        @test all(isapprox(fa[2][ix, iy, iz], fc[2][mod1(ix + 1, Int(M)), iy, iz]; rtol = rtol)
                  for iz in 1:M, iy in 1:M, ix in 1:M)
        @test all(isapprox(fa[3][ix, iy, iz], fc[3][mod1(ix + 1, Int(M)), iy, iz]; rtol = rtol)
                  for iz in 1:M, iy in 1:M, ix in 1:M)
    end
end

@testset "Spread is linear in the input force vector" begin
    # J̃†[αF₁ + βF₂] = α·J̃†[F₁] + β·J̃†[F₂]. Two particles at distinct
    # positions; spread the sum versus the linear combination of separate
    # spreads.
    for T in (Float32, Float64)
        L = (T(16), T(16), T(16))
        M = Int32(16)
        N = 2
        config = FFCMConfig{T}(;
            L = L, R_c = T(1), N = N,
            Σ_over_σ = T(2),
            num_grid_points = (M, M, M),
            M_G = 8,
        )
        config.Y_sorted[1, 1] = T(5); config.Y_sorted[2, 1] = T(6); config.Y_sorted[3, 1] = T(7)
        config.Y_sorted[1, 2] = T(9); config.Y_sorted[2, 2] = T(10); config.Y_sorted[3, 2] = T(11)
        α, β = T(1.7), -T(2.3)
        F1 = (T(1), T(0), T(0))
        F2 = (T(0), T(1), T(2))

        function spread_with(F1, F2)
            config.F_sorted[1, 1] = F1[1]; config.F_sorted[2, 1] = F1[2]; config.F_sorted[3, 1] = F1[3]
            config.F_sorted[1, 2] = F2[1]; config.F_sorted[2, 2] = F2[2]; config.F_sorted[3, 2] = F2[3]
            spread_forces!(config)
            fx, fy, fz = components(config.force_grid)
            return (copy(fx), copy(fy), copy(fz))
        end

        f1 = spread_with(F1, (T(0), T(0), T(0)))
        f2 = spread_with((T(0), T(0), T(0)), F2)
        f_combined = spread_with(
            (α * F1[1] + β * T(0), α * F1[2] + β * T(0), α * F1[3] + β * T(0)),
            (α * T(0) + β * F2[1], α * T(0) + β * F2[2], α * T(0) + β * F2[3]),
        )
        rtol = sqrt(eps(T))
        @test all(isapprox(f_combined[1][i], α * f1[1][i] + β * f2[1][i]; rtol = rtol)
                  for i in eachindex(f1[1]))
        @test all(isapprox(f_combined[2][i], α * f1[2][i] + β * f2[2][i]; rtol = rtol)
                  for i in eachindex(f1[2]))
        @test all(isapprox(f_combined[3][i], α * f1[3][i] + β * f2[3][i]; rtol = rtol)
                  for i in eachindex(f1[3]))
    end
end

@testset "Particle exactly on a grid point produces a reflection-symmetric spread" begin
    # The kernel is even in `x - Yₙ` along each axis; placing the
    # particle on a grid point makes the discrete stencil symmetric.
    for T in (Float32, Float64)
        L = (T(8), T(8), T(8))
        M = Int32(16)
        M_G = 9
        config = FFCMConfig{T}(;
            L = L, R_c = T(1), N = 1,
            Σ_over_σ = T(2),
            num_grid_points = (M, M, M),
            M_G = M_G,
        )
        Δx = config.Δx
        # `Y = 4·Δx` is the grid point at 1-based index 5 in each axis.
        anchor = Int32(4)
        Y_val = T(anchor) * Δx
        config.Y_sorted[1, 1] = Y_val
        config.Y_sorted[2, 1] = Y_val
        config.Y_sorted[3, 1] = Y_val
        config.F_sorted[1, 1] = T(1)
        config.F_sorted[2, 1] = T(0)
        config.F_sorted[3, 1] = T(0)
        spread_forces!(config)

        fx = components(config.force_grid)[1]
        rtol = sqrt(eps(T))
        # Reflect over the anchor (1-based index `anchor + 1`) on axis x.
        for δ in 1:((M_G - 1) ÷ 2)
            i_lo = anchor + 1 - δ
            i_hi = anchor + 1 + δ
            @test fx[i_lo, anchor + 1, anchor + 1] ≈ fx[i_hi, anchor + 1, anchor + 1] rtol=rtol
            @test fx[anchor + 1, i_lo, anchor + 1] ≈ fx[anchor + 1, i_hi, anchor + 1] rtol=rtol
            @test fx[anchor + 1, anchor + 1, i_lo] ≈ fx[anchor + 1, anchor + 1, i_hi] rtol=rtol
        end
    end
end

@testset "Σ = σ collapses the modified kernel to the standard-FCM Gaussian" begin
    # `Σ_over_σ = 1` ⇒ pdmag = 0 ⇒ a₀ = 1, a₂ = 0 ⇒ Δ̃ₙ = Δₙ.
    # The particle sits at the box centre so the M_G^3 stencil never wraps
    # around the periodic boundary; the impl's unwrapped stencil distance
    # then coincides with the canonical grid-point distance and the spread
    # matches the closed-form plain Gaussian at every grid point. Off-centre
    # particles also produce a correct standard-FCM spread, but for grid
    # points reached only via the wrap the impl uses the n = ±1 periodic
    # image's distance (same as cuFCM), which differs from the n = 0
    # closed-form Gaussian — a known artefact of truncated-stencil periodic
    # spreading, not relevant to this collapse test.
    for T in (Float32, Float64)
        L = (T(8), T(8), T(8))
        M = Int32(16)
        config = FFCMConfig{T}(;
            L = L, R_c = T(1), N = 1,
            Σ_over_σ = T(1),
            num_grid_points = (M, M, M),
            M_G = 16,
        )
        Y_n = (T(4), T(4), T(4))   # box centre — no periodic wrap of the stencil
        config.Y_sorted[1, 1] = Y_n[1]
        config.Y_sorted[2, 1] = Y_n[2]
        config.Y_sorted[3, 1] = Y_n[3]
        config.F_sorted[1, 1] = T(1)
        config.F_sorted[2, 1] = T(0)
        config.F_sorted[3, 1] = T(0)
        spread_forces!(config)

        σ = config.σ
        Δx = config.Δx
        fx = components(config.force_grid)[1]
        rtol = sqrt(eps(T))
        for iz in 1:M, iy in 1:M, ix in 1:M
            x_g = ((ix - 1) * Δx, (iy - 1) * Δx, (iz - 1) * Δx)
            r² = (x_g[1] - Y_n[1])^2 + (x_g[2] - Y_n[2])^2 + (x_g[3] - Y_n[3])^2
            Δ = (T(2) * T(π) * σ^2)^(-T(3)/T(2)) * exp(-r²/(T(2) * σ^2))
            @test fx[ix, iy, iz] ≈ Δ rtol=rtol
        end
    end
end
