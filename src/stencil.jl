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
