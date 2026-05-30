using Test
using FFCM
using FFCM: wrap_positions!, assign_cells!, sort_particles_by_cell!,
    spread_forces!, stokes_solve!, interpolate_velocities!, correct_velocities!

# Reciprocal-lattice sum of the σ-regularised periodic Stokeslet self-mobility
# (paper eq 205-207, Fourier form), the true M^VF self-mobility a single particle
# feels from its own periodic images. Independent of Σ.
function _periodic_self_mobility_xx(σ::T, μ::T, Lx::T; n_max::Int = 15) where {T}
    s = zero(T)
    for nx in -n_max:n_max, ny in -n_max:n_max, nz in -n_max:n_max
        (nx == 0 && ny == 0 && nz == 0) && continue
        kx = T(2π * nx / Lx)
        ky = T(2π * ny / Lx)
        kz = T(2π * nz / Lx)
        k² = kx * kx + ky * ky + kz * kz
        s += exp(-σ^2 * k²) / (μ * k²) * (one(T) - kx * kx / k²)
    end
    return s / Lx^3
end

@testset "Single-sphere periodic self-mobility (end-to-end, all five steps)" begin
    # Compose steps 1–5 (wrap → hash → sort → spread → solve → interpolate)
    # for one particle with unit x-force in the standard-FCM degenerate
    # limit (Σ/σ = 1). The interpolated self-velocity is the periodic FCM
    # self-mobility, whose closed form is the reciprocal-lattice sum of the
    # Gaussian-regularised periodic Stokeslet (paper eq 205–207, the
    # Stokeslet convolved with the kernel on both the spread and the
    # interpolate side):
    #
    #   Ṽ_self = (1/L³) Σ_{k≠0} e^{-σ²k²}/(μ k²) (I − k̂k̂ᵀ) · F,
    #   k = (2π/L) n,  n ∈ ℤ³ \ {0}.
    #
    # This continuum sum captures every periodic finite-size correction
    # exactly; our discrete operator approximates it, with error set by the
    # grid resolution (σ/Δx ≈ 2.26 here) and the M_G³ stencil truncation.
    # The single test exercising all five steps composed.
    T = Float64
    L = (T(8), T(8), T(8))
    M = Int32(32)
    config = FFCMConfig{T}(;
        L = L, R_c = T(1), N = 1,
        Σ_over_σ = T(1),
        μ = T(1),
        num_grid_points = (M, M, M),
        # M_G = 21 (odd ⇒ symmetric stencil) reaches (M_G/2)Δx ≈ 4.65σ, so
        # the Gaussian is captured to ~2e-5 relative at the stencil edge.
        M_G = 21,
    )
    Y = T[L[1] / 2; L[2] / 2; L[3] / 2;;]
    F = T[1; 0; 0;;]
    wrap_positions!(Y, config.L)
    assign_cells!(config, Y)
    sort_particles_by_cell!(config, Y, F)
    spread_forces!(config)
    stokes_solve!(config)
    V = zeros(T, 3, 1)
    interpolate_velocities!(V, config)

    σ, μ, Lx = config.σ, config.μ, L[1]
    # n_max = 15 ⇒ e^{-σ²k²} ≈ e^{-44} at the truncation edge — far below
    # round-off, so the lattice sum is converged.
    n_max = 15
    self_mobility_xx = zero(T)
    for nx in -n_max:n_max, ny in -n_max:n_max, nz in -n_max:n_max
        (nx == 0 && ny == 0 && nz == 0) && continue
        kx = T(2π * nx / Lx)
        ky = T(2π * ny / Lx)
        kz = T(2π * nz / Lx)
        k² = kx * kx + ky * ky + kz * kz
        self_mobility_xx += exp(-σ^2 * k²) / (μ * k²) * (one(T) - kx * kx / k²)
    end
    self_mobility_xx /= Lx^3

    # Grid-resolution / stencil-truncation limited (≈ 9e-7 at this σ/Δx),
    # not round-off; rtol = 1e-5 holds with ~10× margin.
    rtol = T(1e-5)
    @test V[1, 1] ≈ self_mobility_xx rtol = rtol

    # Isotropy: an x-force produces motion only along x; the transverse
    # self-mobility vanishes by symmetry.
    @test isapprox(V[2, 1], zero(T); atol = T(1e-12))
    @test isapprox(V[3, 1], zero(T); atol = T(1e-12))

    # Physical sanity: the periodic self-mobility is below the unbounded
    # Stokes value 1/(6πμa) because the periodic images drag the particle
    # back (a = σ√π = 1 here).
    @test zero(T) < V[1, 1] < one(T) / (T(6π) * μ * one(T))
end

@testset "Σ-independent single-sphere self-mobility (end-to-end, all six steps)" begin
    # The whole point of the splitting M = M̃ + (M − M̃): the assembled six-step
    # operator returns the true σ-regularised self-mobility, *independent of the
    # modified-kernel width Σ*. For a single particle the only correction is the
    # self term (no neighbour within R_c; the periodic images at distance L ≫ R_c
    # contribute a correction ~exp(−L²/4Σ²) ≈ 1e-13 here, below grid error and
    # consistent with the paper's large-box assumption, outline.tex:318).
    #
    # The grid step alone (steps 3-5) computes M̃ at width Σ; adding the self term
    # (step 6) must recover the σ-regularised reciprocal-lattice sum — the same
    # reference the Σ = 1 step-5 test uses — for every Σ.
    T = Float64
    L = (T(8), T(8), T(8))
    M = Int32(32)
    # M_G = 31 (odd ⇒ symmetric) captures the Σ-Gaussian to ~5e-6 tail mass even at
    # Σ/σ = 1.5 ((M_G/2)Δx ≈ 4.6Σ); M_G < M = 32 so the stencil does not self-wrap.
    M_G = 31

    σ = one(T) / sqrt(T(π))
    reference = _periodic_self_mobility_xx(σ, T(1), L[1])

    Y0 = T[L[1] / 2; L[2] / 2; L[3] / 2;;]
    F = T[1; 0; 0;;]

    velocities = T[]
    for Σ_over_σ in (T(1.25), T(1.5))
        config = FFCMConfig{T}(;
            L = L, R_c = T(1), N = 1, Σ_over_σ = Σ_over_σ, μ = T(1),
            num_grid_points = (M, M, M), M_G = M_G,
        )
        Y = copy(Y0)
        wrap_positions!(Y, config.L)
        assign_cells!(config, Y)
        sort_particles_by_cell!(config, Y, F)
        spread_forces!(config)
        stokes_solve!(config)
        V = zeros(T, 3, 1)
        interpolate_velocities!(V, config)
        correct_velocities!(V, config)

        # Grid-resolution / stencil-truncation limited; rtol = 1e-4 holds with margin.
        @test V[1, 1] ≈ reference rtol = T(1e-4)
        @test isapprox(V[2, 1], zero(T); atol = T(1e-10))
        @test isapprox(V[3, 1], zero(T); atol = T(1e-10))
        push!(velocities, V[1, 1])
    end

    # Σ-independence: the two corrected self-velocities agree with each other.
    @test velocities[1] ≈ velocities[2] rtol = T(1e-4)
end
