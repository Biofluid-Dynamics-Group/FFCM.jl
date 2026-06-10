# Shared helpers for the test suite. Included once by runtests.jl before any
# test file, so every helper here is visible to all tests.

using FFCM

# Absolute tolerance for quantities that should be ≈ 0, where a relative
# tolerance is ill-defined (CLAUDE.md §2 rule 5): atol ≈ rtol/100 with
# rtol = sqrt(eps(T)).
_near_zero_atol(::Type{Float32}) = 1.0f-6
_near_zero_atol(::Type{Float64}) = 1.0e-10

# Picks `Σ_over_σ`, `num_grid_points`, and `M_G` for tests that only exercise
# the step-1 / step-2 cell-list machinery and do not care about the FCM grid.
# `num_grid_points` is sized so h is isotropic across axes (paper §3
# assumption); a target h of `L[1] / 8` works for every L the existing tests
# use.
function _fcm_grid_kwargs(L::NTuple{3, T}) where {T}
    h_target = L[1] / 8
    num_grid_points = ntuple(i -> Int32(round(Int, L[i] / h_target)), 3)
    return (
        Σ_over_σ = T(2),
        num_grid_points = num_grid_points,
        # The stencil cannot exceed the grid on any axis; clamp for thin domains
        # where a short axis yields fewer than 8 grid points.
        M_G = min(8, Int(minimum(num_grid_points))),
        μ = T(1),
    )
end

# A small multi-particle configuration shared by the step and operator tests.
# Grid and kernel widths follow the regime the accuracy tests use (h isotropic,
# M_G < M); override any keyword for a test that needs a different geometry.
function _standard_test_config(
    ::Type{T};
    N,
    L = (T(8), T(8), T(8)),
    num_grid_points = (Int32(16), Int32(16), Int32(16)),
    M_G = 8,
    Σ_over_σ = T(2),
    μ = T(1),
    R_c = T(1),
) where {T}
    return FFCMConfig{T}(;
        L = L, R_c = R_c, N = N, Σ_over_σ = Σ_over_σ, μ = μ,
        num_grid_points = num_grid_points, M_G = M_G,
    )
end

# Positions and forces for a compact particle cluster around the box centre:
# spacing 0.25 on a 4×4×4 sub-lattice, so many pairs fall within R_c = 1 and
# the pair-correction branch is exercised. Forces are a deterministic
# non-trivial pattern.
function _clustered_cloud(::Type{T}, N) where {T}
    Y = Matrix{T}(undef, 3, N)
    F = Matrix{T}(undef, 3, N)
    for n in 1:N
        Y[1, n] = T(4) + T(0.25) * ((n - 1) % 4)
        Y[2, n] = T(4) + T(0.25) * ((n - 1) ÷ 4 % 4)
        Y[3, n] = T(4) + T(0.25) * ((n - 1) ÷ 16)
        F[1, n] = T(0.1) * n
        F[2, n] = T(-0.2) * n
        F[3, n] = T(0.3) * n
    end
    return Y, F
end

# Fills the config's sorted-particle buffers directly (bypassing steps 1–2)
# with every particle at the box centre and a deterministic force pattern —
# sufficient input for spread/interpolate API checks, where positions only
# need to be in-domain.
function _fill_sorted_midbox!(config, L::NTuple{3, T}, N) where {T}
    for n in 1:N
        for i in 1:3
            config.Y_sorted[i, n] = L[i] * T(0.5)
            config.F_sorted[i, n] = T(0.1) * n
        end
        config.original_index[n] = Int32(n)
    end
    return config
end
