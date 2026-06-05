using Test
using FFCM
using FFCM: stokes_solve!, _apply_inverse_stokes_kernel!,
    spread_forces!, sort_particles_by_cell!, assign_cells!, wrap_positions!
using LinearAlgebra: mul!
using StaticArrays
using StructArrays: components

# Cold-path boilerplate sufficient for every step-4 test below.
const _STOKES_KWARGS = (;
    L = (4.0, 4.0, 4.0),
    R_c = 1.0,
    N = 1,
    Σ_over_σ = 2.0,
    num_grid_points = (Int32(8), Int32(8), Int32(8)),
    M_G = 8,
    μ = 1.0,
)

@testset "Viscosity μ is a required, positive cold-path parameter (paper eq 152)" begin
    # Paper writes the viscosity as η (eq 152, outline.tex:152); this
    # package writes μ (see `spec/stokes-solve.md` notation note). The
    # parameter is mandatory because viscosity is dimensionally meaningful
    # and a wrong default would silently scale every velocity in the
    # downstream mobility application.
    config = FFCMConfig{Float64}(; _STOKES_KWARGS..., μ = 2.5)
    @test config.μ == 2.5

    @test_throws ArgumentError FFCMConfig{Float64}(; _STOKES_KWARGS..., μ = 0.0)
    @test_throws ArgumentError FFCMConfig{Float64}(; _STOKES_KWARGS..., μ = -1.0)
end

@testset "Velocity grid mirrors the force grid: SoA StructArray of shape (M_x, M_y, M_z)" begin
    # Step 5 (interpolation, future) will read fluid_velocity[i_x, i_y, i_z]
    # as an SVector{3, T} exactly as step 3 writes force_density. The cold-path
    # layout pins that interface.
    M = (Int32(8), Int32(12), Int32(16))
    L = (4.0, 6.0, 8.0)
    config = FFCMConfig{Float64}(; _STOKES_KWARGS..., L = L, num_grid_points = M)
    @test config.fluid_velocity isa StructArray
    @test size(config.fluid_velocity) == (8, 12, 16)
    @test eltype(config.fluid_velocity) == SVector{3, Float64}
    comps = components(config.fluid_velocity)
    @test length(comps) == 3
    @test all(c -> c isa Array{Float64, 3}, comps)
    @test all(c -> size(c) == (8, 12, 16), comps)
end

@testset "Fluid-domain buffer fluid_hat has the r2c half-axis layout" begin
    # FFTW's r2c forward halves the leading dimension: a real (M_x, M_y, M_z)
    # array transforms to a complex (M_x÷2+1, M_y, M_z) array. The
    # `fluid_hat` cold-path buffer is sized for this and reused in place by
    # the projection.
    M = (Int32(8), Int32(12), Int32(16))
    L = (4.0, 6.0, 8.0)
    config = FFCMConfig{Float64}(; _STOKES_KWARGS..., L = L, num_grid_points = M)
    @test config.fluid_hat isa StructArray
    @test size(config.fluid_hat) == (8 ÷ 2 + 1, 12, 16)
    @test eltype(config.fluid_hat) == SVector{3, ComplexF64}
    comps = components(config.fluid_hat)
    @test length(comps) == 3
    @test all(c -> c isa Array{ComplexF64, 3}, comps)
    @test all(c -> size(c) == (8 ÷ 2 + 1, 12, 16), comps)
end

@testset "Precomputed wavenumbers follow the FFTW r2c / wrap-around layout" begin
    # k_x covers the non-negative r2c half-axis (0, 2π/L_x, 2·2π/L_x, …,
    # (M_x/2)·2π/L_x). k_y and k_z cover the full FFTW wrap-around order:
    # 0, 1, 2, …, M_i/2, then negative -M_i/2+1, …, -1, all scaled by
    # 2π/L_i.
    M = (Int32(8), Int32(12), Int32(16))
    L = (4.0, 6.0, 8.0)
    config = FFCMConfig{Float64}(; _STOKES_KWARGS..., L = L, num_grid_points = M)
    @test length(config.k_x) == M[1] ÷ 2 + 1
    @test length(config.k_y) == M[2]
    @test length(config.k_z) == M[3]

    PI2 = 2π
    @test config.k_x[1] ≈ 0
    @test config.k_x[2] ≈ PI2 / L[1]
    @test config.k_x[end] ≈ PI2 / L[1] * (M[1] ÷ 2)

    @test config.k_y[1] ≈ 0
    @test config.k_y[2] ≈ PI2 / L[2]
    @test config.k_y[M[2] ÷ 2 + 1] ≈ PI2 / L[2] * (M[2] ÷ 2)
    # First negative wavenumber: index M_y/2 + 2 → physical n = -M_y/2 + 1
    @test config.k_y[M[2] ÷ 2 + 2] ≈ PI2 / L[2] * (-(M[2] ÷ 2) + 1)
    @test config.k_y[end] ≈ PI2 / L[2] * (-1)

    @test config.k_z[1] ≈ 0
    @test config.k_z[M[3] ÷ 2 + 1] ≈ PI2 / L[3] * (M[3] ÷ 2)
    @test config.k_z[end] ≈ PI2 / L[3] * (-1)
end

@testset "FFTW plans are cold-path fields ready for hot-path mul!" begin
    # The plans are stored once at construction so the hot path runs
    # `mul!(out, plan, in)` allocation-free. Their concrete types live in
    # FFCMConfig's type parameters; failing to specialise here would
    # surface as a type-instability downstream.
    config = FFCMConfig{Float64}(; _STOKES_KWARGS...)
    @test config.forward_fourier_transform !== nothing
    @test config.inverse_fourier_transform !== nothing
end

@testset "Cold-path validation works for Float32 as well as Float64" begin
    config = FFCMConfig{Float32}(;
        L = (4.0f0, 4.0f0, 4.0f0),
        R_c = 1.0f0,
        N = 1,
        Σ_over_σ = 2.0f0,
        num_grid_points = (Int32(8), Int32(8), Int32(8)),
        M_G = 8,
        μ = 1.5f0,
    )
    @test config.μ == 1.5f0
    @test eltype(config.fluid_velocity) == SVector{3, Float32}
    @test eltype(config.fluid_hat) == SVector{3, ComplexF32}
    @test eltype(config.k_x) == Float32
end

@testset "Mean velocity is exactly zero (k = 0 gauge fix)" begin
    # Periodic Stokes is undefined at k = 0; the mean velocity is gauge-
    # fixed to zero (spec/stokes-solve.md > Mean-flow gauge fix). The
    # discrete identity Σ_x u(x) = û(k = 0) then implies that the sum of
    # each velocity component over the grid is zero to round-off,
    # regardless of whether the input force has zero mean (the gauge fix
    # silently absorbs any mean component of the input).
    for T in (Float32, Float64)
        config = FFCMConfig{T}(;
            L = (T(4), T(4), T(4)), R_c = T(1), N = 1,
            Σ_over_σ = T(2),
            num_grid_points = (Int32(8), Int32(8), Int32(8)),
            M_G = 8, μ = T(1),
        )
        fx, fy, fz = components(config.force_density)
        # Deterministic non-mean-zero force field.
        for iz in axes(fx, 3), iy in axes(fx, 2), ix in axes(fx, 1)
            fx[ix, iy, iz] = T(ix) + T(iy) * T(0.3) - T(iz) * T(0.5) + T(1)
            fy[ix, iy, iz] = sin(T(ix) + T(iy)) * T(2) + T(0.7)
            fz[ix, iy, iz] = cos(T(ix) * T(iz)) * T(0.5) + T(0.3)
        end
        stokes_solve!(config)
        ux, uy, uz = components(config.fluid_velocity)
        atol = sqrt(eps(T))
        @test abs(sum(ux)) ≤ atol
        @test abs(sum(uy)) ≤ atol
        @test abs(sum(uz)) ≤ atol
    end
end

@testset "Velocity field is discretely divergence-free (k · û = 0 per Fourier mode)" begin
    # Incompressibility ∇·u = 0 in the continuous problem maps in Fourier
    # space to k · û(k) = 0 at every wavenumber. The projector
    # (I − k̂k̂ᵀ)/(μ k²) enforces this identity exactly per mode. We
    # inspect the Fourier-space velocity straight out of
    # `_apply_inverse_stokes_kernel!` (the projector's output) — the c2r
    # backward FFT step inside `stokes_solve!` destroys this buffer in
    # place, so we cannot read it after a full `stokes_solve!` call.
    # Asserting |k · û_mod| ≤ sqrt(eps(T)) · max(‖û_mod‖, 1) per mode; the
    # lower clamp prevents over-tightening at modes where ‖û_mod‖ itself
    # is O(eps(T)).
    for T in (Float32, Float64)
        config = FFCMConfig{T}(;
            L = (T(4), T(4), T(4)), R_c = T(1), N = 1,
            Σ_over_σ = T(2),
            num_grid_points = (Int32(8), Int32(8), Int32(8)),
            M_G = 8, μ = T(1),
        )
        fx, fy, fz = components(config.force_density)
        for iz in axes(fx, 3), iy in axes(fx, 2), ix in axes(fx, 1)
            fx[ix, iy, iz] = sin(T(ix) * T(0.7)) * T(0.5)
            fy[ix, iy, iz] = cos(T(iy) + T(iz) * T(0.4))
            fz[ix, iy, iz] = T(ix) - T(iy) + T(iz) * T(0.2)
        end

        # Forward FFT + projection only; skip the destructive backward FFT.
        fx̂, fŷ, fẑ = components(config.fluid_hat)
        mul!(fx̂, config.forward_fourier_transform, fx)
        mul!(fŷ, config.forward_fourier_transform, fy)
        mul!(fẑ, config.forward_fourier_transform, fz)
        M = Int(config.num_grid_points[1]) * Int(config.num_grid_points[2]) *
                  Int(config.num_grid_points[3])
        inv_M = one(T) / T(M)
        _apply_inverse_stokes_kernel!(
            fx̂, fŷ, fẑ,
            config.k_x, config.k_y, config.k_z,
            config.μ, inv_M,
        )

        rtol = sqrt(eps(T))
        for iz in eachindex(config.k_z),
            iy in eachindex(config.k_y),
            ix in eachindex(config.k_x)

            kx, ky, kz = config.k_x[ix], config.k_y[iy], config.k_z[iz]
            ûx_v = fx̂[ix, iy, iz]
            ûy_v = fŷ[ix, iy, iz]
            ûz_v = fẑ[ix, iy, iz]
            kdotû = kx * ûx_v + ky * ûy_v + kz * ûz_v
            norm_û = sqrt(abs2(ûx_v) + abs2(ûy_v) + abs2(ûz_v))
            @test abs(kdotû) ≤ rtol * max(norm_û, one(T))
        end
    end
end

@testset "Single Fourier mode along x with y-forcing matches the analytical projection" begin
    # f(x) = sin(2π m x / L_x) ê_y for integer m ∈ (0, M_x/2). Since k is
    # along ê_x and f is along ê_y, k · f = 0 — the projector acts as the
    # identity on the forcing, and û(k) = f̂(k) / (μ k²) at the two
    # conjugate-symmetric nonzero modes, zero elsewhere. Inverting back
    # to real space gives u_y(x) = (1 / (μ k_x²)) sin(2π m x / L_x).
    # This pins both the projector arithmetic and the FFT round-trip
    # normalisation.
    for T in (Float32, Float64)
        L = (T(4), T(4), T(4))
        M_x, M_y, M_z = 8, 8, 8
        config = FFCMConfig{T}(;
            L = L, R_c = T(1), N = 1,
            Σ_over_σ = T(2),
            num_grid_points = (Int32(M_x), Int32(M_y), Int32(M_z)),
            M_G = 8, μ = T(1.5),
        )
        m = 1
        kx_mode = T(2) * T(π) * T(m) / L[1]
        fx, fy, fz = components(config.force_density)
        for iz in 1:M_z, iy in 1:M_y, ix in 1:M_x
            x = T(ix - 1) * config.h
            fx[ix, iy, iz] = zero(T)
            fy[ix, iy, iz] = sin(kx_mode * x)
            fz[ix, iy, iz] = zero(T)
        end
        stokes_solve!(config)
        ux, uy, uz = components(config.fluid_velocity)

        scale = one(T) / (config.μ * kx_mode * kx_mode)
        atol = sqrt(eps(T)) * scale
        for iz in 1:M_z, iy in 1:M_y, ix in 1:M_x
            x = T(ix - 1) * config.h
            expected_y = scale * sin(kx_mode * x)
            @test isapprox(ux[ix, iy, iz], zero(T); atol = atol)
            @test isapprox(uy[ix, iy, iz], expected_y; atol = atol)
            @test isapprox(uz[ix, iy, iz], zero(T); atol = atol)
        end
    end
end

@testset "Stokes solve is linear in the force field" begin
    # L^{-1}[αf₁ + βf₂] = α L^{-1}[f₁] + β L^{-1}[f₂]. The inverse Stokes
    # operator is linear by construction; the test exercises the projection
    # arithmetic and FFT round-trip together against a linear combination.
    for T in (Float32, Float64)
        config = FFCMConfig{T}(;
            L = (T(4), T(4), T(4)), R_c = T(1), N = 1,
            Σ_over_σ = T(2),
            num_grid_points = (Int32(8), Int32(8), Int32(8)),
            M_G = 8, μ = T(1),
        )
        α, β = T(1.7), -T(2.3)

        function solve_with(set_force!)
            fx, fy, fz = components(config.force_density)
            set_force!(fx, fy, fz)
            stokes_solve!(config)
            ux, uy, uz = components(config.fluid_velocity)
            return (copy(ux), copy(uy), copy(uz))
        end

        u1 = solve_with(function (fx, fy, fz)
            for iz in axes(fx, 3), iy in axes(fx, 2), ix in axes(fx, 1)
                fx[ix, iy, iz] = sin(T(ix) + T(iy))
                fy[ix, iy, iz] = cos(T(iz) * T(0.5))
                fz[ix, iy, iz] = T(ix) * T(0.1)
            end
        end)
        u2 = solve_with(function (fx, fy, fz)
            for iz in axes(fx, 3), iy in axes(fx, 2), ix in axes(fx, 1)
                fx[ix, iy, iz] = T(iy) - T(iz) * T(0.3)
                fy[ix, iy, iz] = sin(T(ix) * T(iy) * T(0.2))
                fz[ix, iy, iz] = cos(T(iz) + T(ix) * T(0.4))
            end
        end)
        u_combined = solve_with(function (fx, fy, fz)
            for iz in axes(fx, 3), iy in axes(fx, 2), ix in axes(fx, 1)
                fx[ix, iy, iz] = α * (sin(T(ix) + T(iy))) +
                                 β * (T(iy) - T(iz) * T(0.3))
                fy[ix, iy, iz] = α * (cos(T(iz) * T(0.5))) +
                                 β * (sin(T(ix) * T(iy) * T(0.2)))
                fz[ix, iy, iz] = α * (T(ix) * T(0.1)) +
                                 β * (cos(T(iz) + T(ix) * T(0.4)))
            end
        end)

        rtol = sqrt(eps(T))
        for comp in 1:3, i in eachindex(u1[comp])
            @test isapprox(
                u_combined[comp][i],
                α * u1[comp][i] + β * u2[comp][i];
                atol = rtol,
                rtol = rtol,
            )
        end
    end
end

@testset "Cyclic shift of the force shifts the velocity by the same amount" begin
    # The inverse Stokes operator is a Fourier-space multiplier and
    # therefore commutes with translations on the periodic grid. A
    # one-grid-step cyclic shift of the force along an axis induces the
    # same cyclic shift on the velocity.
    for T in (Float32, Float64)
        config = FFCMConfig{T}(;
            L = (T(4), T(4), T(4)), R_c = T(1), N = 1,
            Σ_over_σ = T(2),
            num_grid_points = (Int32(8), Int32(8), Int32(8)),
            M_G = 8, μ = T(1),
        )

        function solve_with_force(base_force)
            fx, fy, fz = components(config.force_density)
            copyto!(fx, base_force[1])
            copyto!(fy, base_force[2])
            copyto!(fz, base_force[3])
            stokes_solve!(config)
            ux, uy, uz = components(config.fluid_velocity)
            return (copy(ux), copy(uy), copy(uz))
        end

        # Build a deterministic non-trivial force field.
        Mx, My, Mz = 8, 8, 8
        base_fx = T[sin(T(ix) + T(iy)) for ix in 1:Mx, iy in 1:My, iz in 1:Mz]
        base_fy = T[cos(T(iy) * T(iz) * T(0.3)) for ix in 1:Mx, iy in 1:My, iz in 1:Mz]
        base_fz = T[T(ix) - T(iy) * T(0.4) for ix in 1:Mx, iy in 1:My, iz in 1:Mz]
        u_a = solve_with_force((base_fx, base_fy, base_fz))

        # Cyclic shift by one grid step along axis x.
        shifted_fx = circshift(base_fx, (1, 0, 0))
        shifted_fy = circshift(base_fy, (1, 0, 0))
        shifted_fz = circshift(base_fz, (1, 0, 0))
        u_b = solve_with_force((shifted_fx, shifted_fy, shifted_fz))

        rtol = sqrt(eps(T))
        @test u_b[1] ≈ circshift(u_a[1], (1, 0, 0)) rtol = rtol
        @test u_b[2] ≈ circshift(u_a[2], (1, 0, 0)) rtol = rtol
        @test u_b[3] ≈ circshift(u_a[3], (1, 0, 0)) rtol = rtol
    end
end

@testset "Stokes solve is equivariant under axis-aligned reflections" begin
    # A reflection R_x along the x-axis maps a vector field
    # `v(x, y, z)` to the field `R_x[v](x, y, z) = D_x · v(R_x · (x, y, z))`
    # where D_x = diag(-1, 1, 1) flips the x-component. The Stokes
    # equations are reflection-equivariant, so L^{-1}[R_x f] = R_x L^{-1} f.
    # Discretely, indices map under reflection as i ↦ mod1(M_i - i + 2, M_i).
    #
    # Odd grid dimensions are used here so there is no Nyquist mode on any
    # axis. For even M the Nyquist mode (k = π/h) is its own conjugate-
    # symmetric partner on the discrete grid, breaking exact reflection
    # equivariance of the projector at that single mode — a well-known
    # artefact of the r2c FFT layout, not a bug in the projector. The
    # cycle-5 single-Fourier-mode test already exercises the projector
    # arithmetic at a non-Nyquist mode at the documented `sqrt(eps(T))`
    # tolerance.
    for T in (Float32, Float64)
        Mx, My, Mz = 7, 7, 7
        config = FFCMConfig{T}(;
            L = (T(4), T(4), T(4)), R_c = T(1), N = 1,
            Σ_over_σ = T(2),
            num_grid_points = (Int32(Mx), Int32(My), Int32(Mz)),
            M_G = 4, μ = T(1),
        )

        base_fx = T[sin(T(ix) + T(iy) * T(0.4)) for ix in 1:Mx, iy in 1:My, iz in 1:Mz]
        base_fy = T[cos(T(ix) * T(0.5) + T(iz)) for ix in 1:Mx, iy in 1:My, iz in 1:Mz]
        base_fz = T[T(ix) * T(0.1) - T(iy) * T(0.2) + T(iz) * T(0.3)
                    for ix in 1:Mx, iy in 1:My, iz in 1:Mz]

        function solve(fx_, fy_, fz_)
            fx, fy, fz = components(config.force_density)
            copyto!(fx, fx_)
            copyto!(fy, fy_)
            copyto!(fz, fz_)
            stokes_solve!(config)
            ux, uy, uz = components(config.fluid_velocity)
            return (copy(ux), copy(uy), copy(uz))
        end

        refl_ix = (i, M) -> mod1(M - i + 2, M)
        reflected_fx = T[-base_fx[refl_ix(ix, Mx), iy, iz]
                         for ix in 1:Mx, iy in 1:My, iz in 1:Mz]
        reflected_fy = T[base_fy[refl_ix(ix, Mx), iy, iz]
                         for ix in 1:Mx, iy in 1:My, iz in 1:Mz]
        reflected_fz = T[base_fz[refl_ix(ix, Mx), iy, iz]
                         for ix in 1:Mx, iy in 1:My, iz in 1:Mz]

        u_a = solve(base_fx, base_fy, base_fz)
        u_b = solve(reflected_fx, reflected_fy, reflected_fz)

        rtol = sqrt(eps(T))
        for iz in 1:Mz, iy in 1:My, ix in 1:Mx
            ix_r = refl_ix(ix, Mx)
            @test isapprox(
                u_b[1][ix, iy, iz], -u_a[1][ix_r, iy, iz]; atol = rtol, rtol = rtol,
            )
            @test isapprox(
                u_b[2][ix, iy, iz], u_a[2][ix_r, iy, iz]; atol = rtol, rtol = rtol,
            )
            @test isapprox(
                u_b[3][ix, iy, iz], u_a[3][ix_r, iy, iz]; atol = rtol, rtol = rtol,
            )
        end
    end
end

@testset "Doubling the viscosity halves the velocity" begin
    # The inverse Stokes operator scales as 1/μ. Solving with the same
    # forcing at μ₀ and 2μ₀ produces velocity fields related by exactly a
    # factor of 2 — pinning that the μ parameter enters the kernel
    # linearly through the per-mode scalar `α = 1/(μ k² M)`.
    for T in (Float32, Float64)
        common_kwargs = (;
            L = (T(4), T(4), T(4)), R_c = T(1), N = 1,
            Σ_over_σ = T(2),
            num_grid_points = (Int32(8), Int32(8), Int32(8)),
            M_G = 8,
        )
        config_low = FFCMConfig{T}(; common_kwargs..., μ = T(1))
        config_high = FFCMConfig{T}(; common_kwargs..., μ = T(2))

        force_y = T[sin(T(ix) * T(0.3) + T(iy) * T(0.2) + T(iz) * T(0.1))
                    for ix in 1:8, iy in 1:8, iz in 1:8]
        for cfg in (config_low, config_high)
            fx, fy, fz = components(cfg.force_density)
            fill!(fx, zero(T))
            copyto!(fy, force_y)
            fill!(fz, zero(T))
        end
        stokes_solve!(config_low)
        stokes_solve!(config_high)

        rtol = sqrt(eps(T))
        for comp in 1:3, i in eachindex(components(config_low.fluid_velocity)[comp])
            u_lo = components(config_low.fluid_velocity)[comp][i]
            u_hi = components(config_high.fluid_velocity)[comp][i]
            @test isapprox(T(2) * u_hi, u_lo; atol = rtol, rtol = rtol)
        end
    end
end

@testset "End-to-end pipeline: single-particle spread + Stokes solve produces a Stokeslet-shaped velocity field" begin
    # Compose steps 1–4 (wrap, hash, sort, spread, solve) for a single
    # particle in the standard-FCM degenerate limit (Σ/σ = 1). With one
    # particle at the box centre and unit x-force, the resulting velocity
    # field is the periodised regularised Stokeslet
    # `S(x_g - Y; σ√2) · ê_x` (paper eq 205–207). For a relaxed
    # end-to-end check we pin:
    #   - decay of |u| with distance (Stokeslet falls off as 1/r at far
    #     field; we assert near-particle vs far-particle magnitudes),
    #   - axial symmetry along x (u is invariant under (y, z) → (-y, -z)
    #     reflections about the particle),
    #   - the x-component of the velocity at the particle is positive
    #     (force and resulting flow are aligned along x).
    # A tight numerical comparison to the closed-form Stokeslet (with
    # explicit periodic image summation) is documented in the spec as a
    # future enhancement.
    T = Float64
    L = (T(4), T(4), T(4))
    M = Int32(16)
    # Odd M_G keeps the stencil symmetric about the particle's anchor grid
    # point, which is required for exact axial symmetry of the spread.
    config = FFCMConfig{T}(;
        L = L, R_c = T(1), N = 1,
        Σ_over_σ = T(1),
        num_grid_points = (M, M, M),
        M_G = 13, μ = T(1),
    )
    Y = T[L[1] / 2; L[2] / 2; L[3] / 2;;]
    F = T[1; 0; 0;;]
    wrap_positions!(Y, config.L)
    assign_cells!(config, Y)
    sort_particles_by_cell!(config, Y, F)
    spread_forces!(config)
    stokes_solve!(config)

    ux, uy, uz = components(config.fluid_velocity)
    h = config.h
    centre_idx = (Int(M) ÷ 2 + 1, Int(M) ÷ 2 + 1, Int(M) ÷ 2 + 1)

    @test ux[centre_idx...] > zero(T)

    # Stokeslet decays as 1/r; at three grid steps from the particle the
    # x-velocity should be much smaller than at the particle.
    near_idx = centre_idx
    far_idx = (centre_idx[1] + 3, centre_idx[2], centre_idx[3])
    @test abs(ux[far_idx...]) < abs(ux[near_idx...])

    # Axial symmetry: (y, z) → (-y, -z) about the particle's position
    # leaves u_x unchanged. Pick offsets (±1, 0) and (0, ±1) along y, z.
    @test ux[centre_idx[1], centre_idx[2] + 1, centre_idx[3]] ≈
          ux[centre_idx[1], centre_idx[2] - 1, centre_idx[3]] rtol = sqrt(eps(T))
    @test ux[centre_idx[1], centre_idx[2], centre_idx[3] + 1] ≈
          ux[centre_idx[1], centre_idx[2], centre_idx[3] - 1] rtol = sqrt(eps(T))

    # The y- and z-components have odd parity in (y, z) — they vanish on
    # the axial line through the particle (mod periodic-image small
    # contributions).
    rtol_image = T(1e-2)   # periodic image contributions are non-trivial in a small L = 4 box.
    @test isapprox(uy[centre_idx...], zero(T); atol = rtol_image)
    @test isapprox(uz[centre_idx...], zero(T); atol = rtol_image)
end

@testset "Zero forcing produces zero velocity at every grid point" begin
    # `-μ Δu + ∇p = 0` with `∇·u = 0` and periodic boundary conditions
    # admits only the constant solution; with the mean-flow gauge `û(0) = 0`
    # the constant solution is zero. The Fourier-space projection must
    # reproduce this identity to round-off.
    for T in (Float32, Float64)
        config = FFCMConfig{T}(;
            L = (T(4), T(4), T(4)),
            R_c = T(1),
            N = 1,
            Σ_over_σ = T(2),
            num_grid_points = (Int32(8), Int32(8), Int32(8)),
            M_G = 8,
            μ = T(1),
        )
        # `force_density` is zero-initialised at construction.
        stokes_solve!(config)
        ux, uy, uz = components(config.fluid_velocity)
        @test all(iszero, ux)
        @test all(iszero, uy)
        @test all(iszero, uz)
    end
end
