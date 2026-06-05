"""
    _modified_kernel_coefficients(σ, Σ) -> NTuple{4, T}

Closed-form scalars of the modified FCM kernel (Su & Keaveny 2024, §3
equation (22)). The modified kernel `(1 + (σ² − Σ²)/2 · Δ) Δ(x; Σ)` collapses,
once the Laplacian acts on the isotropic Gaussian `Δ(x; Σ)`, to the polynomial
form `(a₀ + a₂·r²)·Δ(x; Σ)`, evaluated through the separable 1-D Gaussian
weight `inv_norm·exp(−x²·inv_2Σ²)` per axis. Force spreading
(`_spread_forces_kernel!`) and velocity interpolation
(`_interpolate_velocities_kernel!`) share these four scalars verbatim, because
interpolation is the exact discrete adjoint of spreading; centralising them
here keeps the eq-(22) expansion in one place. Pure and allocation-free; called
once per kernel invocation, before the per-particle loop.

# Arguments
- `σ::T`: the physical FCM kernel width, `σ = a/√π` for a unit-radius particle.
- `Σ::T`: the (wider) modified-kernel width, `Σ ≥ σ`.

# Returns
- `NTuple{4, T}`: `(a₀, a₂, inv_norm, inv_2Σ²)`, where
  `a₀ = 1 − 3(σ²−Σ²)/(2Σ²)` and `a₂ = (σ²−Σ²)/(2Σ⁴)` are the polynomial
  coefficients, `inv_norm = 1/√(2πΣ²)` the Gaussian normalisation, and
  `inv_2Σ² = 1/(2Σ²)` the exponent scale. At the standard-FCM limit `Σ = σ`
  the prefactor degenerates to `a₀ = 1`, `a₂ = 0` (the unmodified Gaussian).

See `spec/force-spreading.md` and `spec/interpolation.md`.
"""
function _modified_kernel_coefficients(σ::T, Σ::T) where {T}
    Σ² = Σ * Σ
    Σ⁴ = Σ² * Σ²
    σ²_minus_Σ² = σ * σ - Σ²
    a_0 = one(T) - T(3) * σ²_minus_Σ² / (T(2) * Σ²)
    a_2 = σ²_minus_Σ² / (T(2) * Σ⁴)
    inv_norm = one(T) / sqrt(T(2) * T(π) * Σ²)
    inv_2Σ² = one(T) / (T(2) * Σ²)
    return (a_0, a_2, inv_norm, inv_2Σ²)
end

"""
    _fill_particle_stencil!(
        gaussian_x, gaussian_y, gaussian_z,
        r²_x, r²_y, r²_z,
        idx_x, idx_y, idx_z,
        Y1, Y2, Y3,
        inv_norm, inv_2Σ², h, inv_h, num_grid_points, M_G, half_M_G,
    ) -> nothing

Fill the per-axis stencil scratch for one particle at position
`(Y1, Y2, Y3)`. This is the geometry shared verbatim by force spreading
(`_spread_forces_kernel!`) and velocity interpolation
(`_interpolate_velocities_kernel!`): because interpolation is the exact
discrete adjoint of spreading, both must place and weight the `M_G³` stencil
identically. Keeping the convention in one function is what guarantees that.

For each axis the stencil is nearest-anchored at
`j_i = round(Y_i · inv_h)` (`RoundNearestTiesToEven`, the cuFCM `my_rint`
convention; paper §5 Table 2 calibration). For `k ∈ 1:M_G` it writes:

- `gaussian_*[k] = inv_norm · exp(−x² · inv_2Σ²)` — the separable 1-D Gaussian
  weight, with `x = (j_i − ⌊M_G/2⌋ + (k−1))·h − Y_i` the unwrapped stencil
  distance and `inv_norm = 1/√(2πΣ²)`, `inv_2Σ² = 1/(2Σ²)`.
- `r²_*[k] = x²` — the axis-squared distance (stored directly; the
  modified-kernel polynomial only needs `r²`).
- `idx_*[k]` — the periodic-wrapped 1-based grid index `mod(g_i, M_i) + 1`.

Preconditions (caller-guaranteed, so the loop is `@inbounds`): the nine
scratch vectors have length `M_G`; `half_M_G = M_G ÷ 2`; `num_grid_points` holds
`(M_x, M_y, M_z)`; the position has been folded into `[0, L_i)` by
`wrap_positions!`.

See `spec/force-spreading.md` and `spec/interpolation.md`.
"""
function _fill_particle_stencil!(
    gaussian_x::Vector{T},
    gaussian_y::Vector{T},
    gaussian_z::Vector{T},
    r²_x::Vector{T},
    r²_y::Vector{T},
    r²_z::Vector{T},
    idx_x::Vector{Int32},
    idx_y::Vector{Int32},
    idx_z::Vector{Int32},
    Y1::T,
    Y2::T,
    Y3::T,
    inv_norm::T,
    inv_2Σ²::T,
    h::T,
    inv_h::T,
    num_grid_points::NTuple{3, Int32},
    M_G::Int32,
    half_M_G::Int32,
) where {T}
    M_x, M_y, M_z = num_grid_points
    j1 = round(Int32, Y1 * inv_h)
    j2 = round(Int32, Y2 * inv_h)
    j3 = round(Int32, Y3 * inv_h)

    @inbounds for k in Int32(1):M_G
        offset = k - Int32(1) - half_M_G
        g1 = j1 + offset
        g2 = j2 + offset
        g3 = j3 + offset
        x1 = T(g1) * h - Y1
        x2 = T(g2) * h - Y2
        x3 = T(g3) * h - Y3
        gaussian_x[k] = inv_norm * exp(-x1 * x1 * inv_2Σ²)
        gaussian_y[k] = inv_norm * exp(-x2 * x2 * inv_2Σ²)
        gaussian_z[k] = inv_norm * exp(-x3 * x3 * inv_2Σ²)
        r²_x[k] = x1 * x1
        r²_y[k] = x2 * x2
        r²_z[k] = x3 * x3
        idx_x[k] = mod(g1, M_x) + Int32(1)
        idx_y[k] = mod(g2, M_y) + Int32(1)
        idx_z[k] = mod(g3, M_z) + Int32(1)
    end
    return nothing
end
