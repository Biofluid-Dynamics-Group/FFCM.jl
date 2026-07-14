"""
    _modified_kernel_coefficients(σ, Σ) -> NTuple{4, T}

Closed-form scalars of the modified FCM kernel (Su & Keaveny 2024, §3 equation (22)). The
modified kernel `(1 + (σ² - Σ²)/2 ⋅ Δ) Δ(x; Σ)` collapses, once the Laplacian acts on the
isotropic Gaussian `Δ(x; Σ)`, to the polynomial form `(a₀ + a₂⋅r²)⋅Δ(x; Σ)`, evaluated
through the separable 1-D Gaussian weight `inv_norm⋅exp(-x²⋅inv_2Σ²)` per axis. Spreading
(`_spread_forces_kernel!`) and interpolation (`_interpolate_velocities_kernel!`) must weight
the stencil identically — interpolation is the discrete adjoint of spreading — so both read
these four scalars.

# Arguments
- `σ::T`: the physical FCM kernel width, `σ = a/√π` for a unit-radius particle.
- `Σ::T`: the (wider) modified-kernel width, `Σ ≥ σ`.

# Returns
- `NTuple{4, T}`: `(a₀, a₂, inv_norm, inv_2Σ²)`, where `a₀ = 1 - 3(σ²−Σ²)/(2Σ²)` and
  `a₂ = (σ²−Σ²)/(2Σ⁴)` are the polynomial coefficients, `inv_norm = 1/√(2πΣ²)` the Gaussian
  normalisation, and `inv_2Σ² = 1/(2Σ²)` the exponent scale. At the standard FCM limit
  `Σ = σ` the prefactor degenerates to `a₀ = 1`, `a₂ = 0` (the unmodified Gaussian).
"""
function _modified_kernel_coefficients(σ::T, Σ::T) where {T}
    Σ² = Σ^2
    Σ⁴ = Σ²^2
    σ²_minus_Σ² = σ^2 - Σ²
    a₀ = one(T) - T(3) * σ²_minus_Σ² / (T(2) * Σ²)
    a₂ = σ²_minus_Σ² / (T(2) * Σ⁴)
    inv_norm = one(T) / sqrt(T(2) * T(π) * Σ²)
    inv_2Σ² = one(T) / (T(2) * Σ²)
    return (a₀, a₂, inv_norm, inv_2Σ²)
end

"""
    _fill_particle_stencil!(
        stencil_gaussian, stencil_r², stencil_index,
        Y1, Y2, Y3,
        inv_norm, inv_2Σ², h, inv_h, num_grid_points, M_G, half_M_G,
    ) -> nothing

Fill the per-particle stencil scratch for one particle at position `(Y1, Y2, Y3)`.
Spreading (`_spread_forces_kernel!`) and interpolation (`_interpolate_velocities_kernel!`)
both place and weight the `M_G³` stencil through this function — interpolation is the
discrete adjoint of spreading, so the geometry must be identical.

For each axis the stencil is nearest-anchored at `j_i = round(Y_i ⋅ inv_h)`
(`RoundNearestTiesToEven`; the nearest-grid-point anchoring the paper's §5 Table 1
calibration assumes).
For `k ∈ 1:M_G` it writes, with `x_i = (j_i - ⌊M_G/2⌋ + (k-1))⋅h - Y_i` the unwrapped
per-axis stencil distance:

- `stencil_gaussian[k]` — the separable 1-D Gaussian weights
  `inv_norm ⋅ exp(-x_i² ⋅ inv_2Σ²)`.
- `stencil_r²[k]` — the axis-squared distances `x_i²` (the modified-kernel polynomial
  needs only `r²`).
- `stencil_index[k]` — the periodic-wrapped 1-based grid indices `mod(g_i, M_i) + 1`.

# Arguments
- `stencil_gaussian`: the `StructArray{SVector{3, T}}` of per-axis Gaussian weights
  (length `M_G`) to overwrite.
- `stencil_r²`: the `StructArray{SVector{3, T}}` of per-axis squared distances (length
  `M_G`).
- `stencil_index`: the `StructArray{SVector{3, Int32}}` of per-axis periodic-wrapped
  1-based grid indices (length `M_G`).
- `Y1::T`, `Y2::T`, `Y3::T`: the particle position, folded into `[0, L_i)`.
- `inv_norm::T`: the Gaussian normalisation `1/√(2πΣ²)`.
- `inv_2Σ²::T`: the Gaussian exponent scale `1/(2Σ²)`.
- `h::T`, `inv_h::T`: the grid spacing and its inverse.
- `num_grid_points::NTuple{3, Int32}`: the grid dimensions `(M_x, M_y, M_z)`.
- `M_G::Int32`: the cubic stencil support per axis.
- `half_M_G::Int32`: `M_G ÷ 2`, the stencil half-width offset.

# Returns
- `nothing`. The three scratch fields are overwritten in place.

# Notes
Preconditions (caller-guaranteed, so the loop is `@inbounds`): the three scratch fields
have length `M_G`; `half_M_G = M_G ÷ 2`; `num_grid_points` holds `(M_x, M_y, M_z)`; the
position has been folded into `[0, L_i)` by `wrap_positions!`.
"""
function _fill_particle_stencil!(
    stencil_gaussian,
    stencil_r²,
    stencil_index,
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
    Y = SVector(Y1, Y2, Y3)
    M = SVector(num_grid_points)
    j = round.(Int32, Y .* inv_h)

    # Each whole-SVector store lowers to the three per-axis component writes
    # of the backing StructArray (benchmarked at parity with writing the
    # components explicitly).
    @inbounds for k in Int32(1):M_G
        offset = k - Int32(1) - half_M_G
        g = j .+ offset
        x = T.(g) .* h .- Y
        x² = x .^ 2
        stencil_gaussian[k] = inv_norm .* exp.(-x² .* inv_2Σ²)
        stencil_r²[k] = x²
        stencil_index[k] = mod.(g, M) .+ Int32(1)
    end
    return nothing
end
