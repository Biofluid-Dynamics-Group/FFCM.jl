using Test
using FFCM
using FFCM: spread_forces!, stokes_solve!, interpolate_velocities!,
    wrap_positions!, assign_cells!, sort_particles_by_cell!
using StaticArrays
using StructArrays: components

@testset "Interpolation is the discrete adjoint of spreading (J = h³ Sᵀ)" begin
    # The mobility M^VF = J·L⁻¹·J† is positive-definite only because the
    # interpolation operator J is the exact discrete transpose of the
    # spreading operator J†, scaled by the trapezoidal weight h³
    # (paper §3). Concretely, for any grid field `u` and
    # any particle/component unit vector `e_{n,c}`:
    #
    #   ⟨interpolate(u), e_{n,c}⟩ = h³ · ⟨u, spread(e_{n,c})⟩.
    #
    # `spread(e_{n,c})` puts a unit force on particle `n` in direction `c`,
    # so force_density component `c` equals Δ̃_n(x_g) and the others vanish;
    # the inner product collapses to a sum of `u_c` against that kernel.
    for T in (Float32, Float64)
        L = (T(8), T(8), T(8))
        M = Int32(16)
        N = 3
        config = FFCMConfig{T}(;
            L = L, R_c = T(1), N = N,
            Σ_over_σ = T(2),
            μ = T(1),
            num_grid_points = (M, M, M),
            M_G = 8,
        )
        Ys = ((T(3.1), T(3.2), T(3.3)),
              (T(4.5), T(4.6), T(4.7)),
              (T(5.2), T(2.7), T(5.4)))
        for n in 1:N, i in 1:3
            config.Y_sorted[i, n] = Ys[n][i]
        end
        config.original_index .= Int32.(1:N)

        ux, uy, uz = components(config.fluid_velocity)
        for (idx, I) in enumerate(CartesianIndices(ux))
            ux[I] = sin(T(0.3) * idx)
            uy[I] = cos(T(0.2) * idx)
            uz[I] = sin(T(0.1) * idx + T(1))
        end

        h³ = config.h^3
        V = zeros(T, 3, N)
        interpolate_velocities!(V, config)

        fx, fy, fz = components(config.force_density)
        u_comp = (ux, uy, uz)
        f_comp = (fx, fy, fz)
        rtol = sqrt(eps(T))
        atol = _near_zero_atol(T)
        for n in 1:N, c in 1:3
            fill!(config.F_sorted, zero(T))
            config.F_sorted[c, n] = one(T)
            spread_forces!(config)
            adjoint_value = h³ * sum(u_comp[c] .* f_comp[c])
            @test V[c, n] ≈ adjoint_value rtol = rtol atol = atol
        end
    end
end

@testset "Interpolated velocities are returned in original particle order" begin
    # interpolate_velocities! writes V[:, original_index[s]], undoing the
    # step-2 cell sort. We place particles so the sort genuinely permutes
    # them, then check each particle's velocity matches an independent
    # single-particle interpolation at that particle's position — i.e. the
    # value landed in the right (original) column.
    for T in (Float32, Float64)
        L = (T(8), T(8), T(8))
        M = Int32(16)
        N = 5
        config = FFCMConfig{T}(;
            L = L, R_c = T(1), N = N,
            Σ_over_σ = T(2),
            μ = T(1),
            num_grid_points = (M, M, M),
            M_G = 8,
        )
        # Decreasing z ⇒ decreasing cell hash ⇒ the counting sort reverses
        # the particle order, so original_index ≠ 1:N.
        Ys = ((T(1.0), T(1.0), T(7.0)),
              (T(2.0), T(2.0), T(6.0)),
              (T(3.0), T(3.0), T(5.0)),
              (T(4.0), T(4.0), T(2.0)),
              (T(5.0), T(5.0), T(1.0)))
        Y = T[Ys[n][i] for i in 1:3, n in 1:N]
        F = zeros(T, 3, N)   # forces are irrelevant to interpolation
        wrap_positions!(Y, config.L)
        assign_cells!(config, Y)
        sort_particles_by_cell!(config, Y, F)
        @test config.original_index != Int32.(1:N)

        deterministic_field! = function (grid)
            gx, gy, gz = components(grid)
            for (idx, I) in enumerate(CartesianIndices(gx))
                gx[I] = sin(T(0.17) * idx)
                gy[I] = cos(T(0.11) * idx + T(0.5))
                gz[I] = sin(T(0.23) * idx - T(0.3))
            end
        end
        deterministic_field!(config.fluid_velocity)

        V = zeros(T, 3, N)
        interpolate_velocities!(V, config)

        ref = FFCMConfig{T}(;
            L = L, R_c = T(1), N = 1,
            Σ_over_σ = T(2),
            μ = T(1),
            num_grid_points = (M, M, M),
            M_G = 8,
        )
        ref.original_index[1] = Int32(1)
        deterministic_field!(ref.fluid_velocity)
        Vn = zeros(T, 3, 1)
        rtol = sqrt(eps(T))
        atol = _near_zero_atol(T)
        for n in 1:N
            ref.Y_sorted[1, 1] = Y[1, n]
            ref.Y_sorted[2, 1] = Y[2, n]
            ref.Y_sorted[3, 1] = Y[3, n]
            interpolate_velocities!(Vn, ref)
            @test isapprox(V[:, n], Vn[:, 1]; rtol = rtol, atol = atol)
        end
    end
end

@testset "Single-particle interpolation matches the closed-form modified kernel" begin
    # One particle at the box centre on a cubic grid whose M_G³ stencil
    # covers every grid point, so the gather sums over the whole grid and
    # can be checked term-by-term against the closed-form modified kernel
    # (paper §3 equation (22), same expansion as the spread test).
    for T in (Float32, Float64)
        L = (T(4), T(4), T(4))
        M = Int32(8)
        config = FFCMConfig{T}(;
            L = L, R_c = T(1), N = 1,
            Σ_over_σ = T(2),
            μ = T(1),
            num_grid_points = (M, M, M),
            M_G = 8,
        )
        Y_n = (T(2), T(2), T(2))
        config.Y_sorted[1, 1] = Y_n[1]
        config.Y_sorted[2, 1] = Y_n[2]
        config.Y_sorted[3, 1] = Y_n[3]
        config.original_index[1] = Int32(1)

        σ, Σ, h = config.σ, config.Σ, config.h
        ux, uy, uz = components(config.fluid_velocity)
        for (idx, I) in enumerate(CartesianIndices(ux))
            ux[I] = sin(T(0.3) * idx)
            uy[I] = cos(T(0.2) * idx)
            uz[I] = T(0.05) * idx - T(1)
        end

        V = zeros(T, 3, 1)
        interpolate_velocities!(V, config)

        h³ = h^3
        ref = (zero(T), zero(T), zero(T))
        for iz in 1:M, iy in 1:M, ix in 1:M
            x_g = ((ix - 1) * h, (iy - 1) * h, (iz - 1) * h)
            r² = (x_g[1] - Y_n[1])^2 + (x_g[2] - Y_n[2])^2 + (x_g[3] - Y_n[3])^2
            Δ = (T(2) * T(π) * Σ^2)^(-T(3) / T(2)) * exp(-r² / (T(2) * Σ^2))
            laplacian = (r² / Σ^4 - T(3) / Σ^2) * Δ
            Δ̃ = Δ + (σ^2 - Σ^2) / T(2) * laplacian
            ref = (
                ref[1] + ux[ix, iy, iz] * Δ̃ * h³,
                ref[2] + uy[ix, iy, iz] * Δ̃ * h³,
                ref[3] + uz[ix, iy, iz] * Δ̃ * h³,
            )
        end
        rtol = sqrt(eps(T))
        atol = _near_zero_atol(T)
        @test V[1, 1] ≈ ref[1] rtol = rtol atol = atol
        @test V[2, 1] ≈ ref[2] rtol = rtol atol = atol
        @test V[3, 1] ≈ ref[3] rtol = rtol atol = atol
    end
end

@testset "Interpolating a constant flow returns that constant (rigid-body limit)" begin
    # ∫ Δ̃_n(x; Σ) d³x = 1 (the Laplacian term integrates to zero), so a
    # spatially constant velocity field interpolates to that same constant
    # at every particle, up to the M_G³ stencil truncation error (paper
    # Table 1). Constant fields are wrap-safe, so particle placement is
    # unconstrained.
    for T in (Float32, Float64)
        L = (T(16), T(16), T(16))
        M = Int32(16)
        N = 4
        config = FFCMConfig{T}(;
            L = L, R_c = T(1), N = N,
            Σ_over_σ = T(2),
            μ = T(1),
            num_grid_points = (M, M, M),
            M_G = 16,
        )
        Ys = ((T(3.3), T(4.1), T(8.0)),
              (T(8.0), T(8.0), T(8.0)),
              (T(11.7), T(2.9), T(13.4)),
              (T(0.4), T(15.6), T(7.2)))
        for n in 1:N, i in 1:3
            config.Y_sorted[i, n] = Ys[n][i]
        end
        config.original_index .= Int32.(1:N)

        U = (T(0.7), -T(1.3), T(2.1))
        ux, uy, uz = components(config.fluid_velocity)
        fill!(ux, U[1])
        fill!(uy, U[2])
        fill!(uz, U[3])

        V = zeros(T, 3, N)
        interpolate_velocities!(V, config)

        # Truncation-dominated; the spread's force-conservation test reaches
        # sqrt(eps(T)) with the same (M_G = 16, Σ/h) regime, so the kernel
        # mass is 1 to that accuracy here too.
        rtol = sqrt(eps(T))
        atol = _near_zero_atol(T)
        expected = T[U[i] for i in 1:3, _ in 1:N]
        @test all(isapprox.(V, expected; rtol = rtol, atol = atol))
    end
end

@testset "Interpolation is linear in the flow field" begin
    # J̃[α u₁ + β u₂] = α·J̃[u₁] + β·J̃[u₂].
    for T in (Float32, Float64)
        L = (T(8), T(8), T(8))
        M = Int32(16)
        N = 2
        config = FFCMConfig{T}(;
            L = L, R_c = T(1), N = N,
            Σ_over_σ = T(2),
            μ = T(1),
            num_grid_points = (M, M, M),
            M_G = 8,
        )
        config.Y_sorted[:, 1] .= (T(3.1), T(3.6), T(4.2))
        config.Y_sorted[:, 2] .= (T(5.5), T(4.4), T(2.9))
        config.original_index .= Int32.(1:N)

        field1!(grid) = begin
            gx, gy, gz = components(grid)
            for (idx, I) in enumerate(CartesianIndices(gx))
                gx[I] = sin(T(0.3) * idx)
                gy[I] = cos(T(0.2) * idx)
                gz[I] = sin(T(0.13) * idx)
            end
        end
        field2!(grid) = begin
            gx, gy, gz = components(grid)
            for (idx, I) in enumerate(CartesianIndices(gx))
                gx[I] = cos(T(0.07) * idx + T(1))
                gy[I] = sin(T(0.29) * idx)
                gz[I] = cos(T(0.19) * idx - T(0.5))
            end
        end

        α, β = T(1.7), -T(2.3)
        ux, uy, uz = components(config.fluid_velocity)

        field1!(config.fluid_velocity)
        V1 = zeros(T, 3, N); interpolate_velocities!(V1, config)
        u1 = (copy(ux), copy(uy), copy(uz))

        field2!(config.fluid_velocity)
        V2 = zeros(T, 3, N); interpolate_velocities!(V2, config)
        u2 = (copy(ux), copy(uy), copy(uz))

        ux .= α .* u1[1] .+ β .* u2[1]
        uy .= α .* u1[2] .+ β .* u2[2]
        uz .= α .* u1[3] .+ β .* u2[3]
        Vc = zeros(T, 3, N); interpolate_velocities!(Vc, config)

        rtol = sqrt(eps(T))
        atol = _near_zero_atol(T)
        @test all(isapprox(Vc[i], α * V1[i] + β * V2[i]; rtol = rtol, atol = atol)
                  for i in eachindex(Vc))
    end
end

@testset "Translating particle and flow by one grid cell leaves the velocity unchanged" begin
    # The gather is periodic and translation-equivariant: shifting the
    # particle by h along axis i and cyclically shifting the flow by one
    # cell on the same axis reproduces the identical stencil reads, so the
    # interpolated velocity is unchanged.
    for T in (Float32, Float64)
        L = (T(8), T(8), T(8))
        M = Int32(16)
        config = FFCMConfig{T}(;
            L = L, R_c = T(1), N = 1,
            Σ_over_σ = T(2),
            μ = T(1),
            num_grid_points = (M, M, M),
            M_G = 8,
        )
        config.original_index[1] = Int32(1)
        h = config.h

        ux, uy, uz = components(config.fluid_velocity)
        base = ntuple(_ -> Array{T}(undef, Int(M), Int(M), Int(M)), 3)
        for (idx, I) in enumerate(CartesianIndices(ux))
            base[1][I] = sin(T(0.3) * idx)
            base[2][I] = cos(T(0.2) * idx)
            base[3][I] = sin(T(0.11) * idx + T(0.7))
        end

        Y_a = (T(3.4), T(4.7), T(5.2))
        ux .= base[1]; uy .= base[2]; uz .= base[3]
        config.Y_sorted[:, 1] .= Y_a
        Va = zeros(T, 3, 1); interpolate_velocities!(Va, config)

        # circshift by +1 along x sends content at ix to ix+1, matching the
        # +1 shift of the stencil indices when the particle moves by +h.
        ux .= circshift(base[1], (1, 0, 0))
        uy .= circshift(base[2], (1, 0, 0))
        uz .= circshift(base[3], (1, 0, 0))
        config.Y_sorted[:, 1] .= (Y_a[1] + h, Y_a[2], Y_a[3])
        Vb = zeros(T, 3, 1); interpolate_velocities!(Vb, config)

        rtol = sqrt(eps(T))
        atol = _near_zero_atol(T)
        @test Va[1, 1] ≈ Vb[1, 1] rtol = rtol atol = atol
        @test Va[2, 1] ≈ Vb[2, 1] rtol = rtol atol = atol
        @test Va[3, 1] ≈ Vb[3, 1] rtol = rtol atol = atol
    end
end

@testset "Σ = σ collapses interpolation to the standard-FCM Gaussian average" begin
    # Σ_over_σ = 1 ⇒ σ²_minus_Σ² = 0 ⇒ a₀ = 1, a₂ = 0 ⇒ Δ̃_n = Δ_n. The particle
    # sits at the box centre and the M_G³ stencil covers the whole grid, so
    # the gather matches the closed-form plain-Gaussian volume average at
    # every grid point.
    for T in (Float32, Float64)
        L = (T(8), T(8), T(8))
        M = Int32(16)
        config = FFCMConfig{T}(;
            L = L, R_c = T(1), N = 1,
            Σ_over_σ = T(1),
            μ = T(1),
            num_grid_points = (M, M, M),
            M_G = 16,
        )
        Y_n = (T(4), T(4), T(4))
        config.Y_sorted[1, 1] = Y_n[1]
        config.Y_sorted[2, 1] = Y_n[2]
        config.Y_sorted[3, 1] = Y_n[3]
        config.original_index[1] = Int32(1)

        σ, h = config.σ, config.h
        ux, uy, uz = components(config.fluid_velocity)
        for (idx, I) in enumerate(CartesianIndices(ux))
            ux[I] = sin(T(0.21) * idx)
            uy[I] = cos(T(0.16) * idx)
            uz[I] = T(0.04) * idx - T(0.5)
        end

        V = zeros(T, 3, 1)
        interpolate_velocities!(V, config)

        h³ = h^3
        ref = (zero(T), zero(T), zero(T))
        for iz in 1:M, iy in 1:M, ix in 1:M
            x_g = ((ix - 1) * h, (iy - 1) * h, (iz - 1) * h)
            r² = (x_g[1] - Y_n[1])^2 + (x_g[2] - Y_n[2])^2 + (x_g[3] - Y_n[3])^2
            Δ = (T(2) * T(π) * σ^2)^(-T(3) / T(2)) * exp(-r² / (T(2) * σ^2))
            ref = (
                ref[1] + ux[ix, iy, iz] * Δ * h³,
                ref[2] + uy[ix, iy, iz] * Δ * h³,
                ref[3] + uz[ix, iy, iz] * Δ * h³,
            )
        end
        rtol = sqrt(eps(T))
        atol = _near_zero_atol(T)
        @test V[1, 1] ≈ ref[1] rtol = rtol atol = atol
        @test V[2, 1] ≈ ref[2] rtol = rtol atol = atol
        @test V[3, 1] ≈ ref[3] rtol = rtol atol = atol
    end
end

@testset "Assembled mobility M^VF is symmetric (spread/solve/interpolate adjoint)" begin
    # M^VF = h³·(SP)ᵀ·L⁻¹·(SP) is symmetric positive-definite because L⁻¹
    # is and interpolation is the transpose of spreading. Symmetry test:
    # Fₐᵀ (M Fᵦ) = Fᵦᵀ (M Fₐ) for arbitrary force vectors.
    for T in (Float32, Float64)
        L = (T(8), T(8), T(8))
        M = Int32(16)
        N = 4
        config = FFCMConfig{T}(;
            L = L, R_c = T(1), N = N,
            Σ_over_σ = T(2),
            μ = T(1),
            num_grid_points = (M, M, M),
            M_G = 8,
        )
        Ys = ((T(2.3), T(3.1), T(4.7)),
              (T(5.5), T(1.9), T(6.2)),
              (T(3.8), T(6.6), T(2.4)),
              (T(6.1), T(4.2), T(5.0)))
        Y = T[Ys[n][i] for i in 1:3, n in 1:N]
        wrap_positions!(Y, config.L)
        assign_cells!(config, Y)

        apply_mobility = function (F)
            sort_particles_by_cell!(config, Y, F)
            spread_forces!(config)
            stokes_solve!(config)
            V = zeros(T, 3, N)
            interpolate_velocities!(V, config)
            return V
        end

        Fa = T[sin(T(n) + T(i)) for i in 1:3, n in 1:N]
        Fb = T[cos(T(2n) - T(i)) for i in 1:3, n in 1:N]
        MFa = apply_mobility(Fa)
        MFb = apply_mobility(Fb)

        rtol = sqrt(eps(T))
        atol = _near_zero_atol(T)
        @test sum(Fa .* MFb) ≈ sum(Fb .* MFa) rtol = rtol atol = atol
    end
end
