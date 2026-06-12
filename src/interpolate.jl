"""
    interpolate_velocities!(V, config) -> V

Step 5 of the Fast FCM algorithm (Su & Keaveny 2024, §4; the interpolation operator of §3
equation (26)). Interpolate the fluid velocity field `config.fluid_velocity` (output of
`stokes_solve!`) to each particle position, evaluating the modified-kernel volume average
`Ṽ_n = ∫ u(x) Δ̃_n(x; Σ) dx` by the trapezoidal rule over the same `M_G³` stencil as the
spread, with weight `h³`.

The scatter-back via `original_index` is the inverse of step 2's gather, which makes
interpolation the exact discrete adjoint of `spread_forces!` and the assembled mobility
operator symmetric positive-definite.

# Arguments
- `V::AbstractMatrix{T}`: a caller-owned `3xN` matrix, overwritten with the particle
  velocities in the caller's **original** particle order.
- `config::FFCMConfig{T}`: the compiled configuration. Reads `config.fluid_velocity`,
  `config.Y_sorted`, and `config.original_index`(populated by `sort_particles_by_cell!`).

# Returns
- `V`: the same matrix, holding the interpolated particle velocities.

See `spec/interpolation.md`.
"""
function interpolate_velocities!(
    V::AbstractMatrix{T}, config::FFCMConfig{T},
) where {T}
    _interpolate_velocities_kernel!(
        V,
        config.fluid_velocity,
        config.Y_sorted,
        config.original_index,
        config.σ,
        config.Σ,
        config.h,
        config.inv_h,
        config.num_grid_points,
        config.M_G,
        config.stencil_gaussian,
        config.stencil_r²,
        config.stencil_index,
    )
    return V
end

"""
    _interpolate_velocities_kernel!(
        V, fluid_velocity, Y_sorted, original_index,
        σ, Σ, h, inv_h, num_grid_points, M_G,
        stencil_gaussian, stencil_r², stencil_index,
    ) -> V

Kernel for `interpolate_velocities!`. For each particle, anchor the stencil at
`j_i = round(Y_{n,i}/h)` and fill the per-axis 1-D Gaussian weights, axis-squared distances,
and periodic-wrapped 1-based stencil indices via `_fill_particle_stencil!` (identical to
`_spread_forces_kernel!`), then gather `u(x_g) ⋅ (a₀ + a₂⋅r²) ⋅ g_x⋅g_y⋅g_z` over the `M_G³`
stencil, scale by `h³`, and write the result to `V[:, original_index[s]]` (the inverse
step-2 permutation). The polynomial coefficients `(a₀, a₂)` and the Gaussian normalisation
come from `_modified_kernel_coefficients` (paper §3 equation (22)).

# Arguments
- `V::AbstractMatrix{T}`: the `3xN` output matrix, written in original order.
- `fluid_velocity`: the `StructArray{SVector{3, T}}` velocity grid field to gather from.
- `Y_sorted::AbstractMatrix{T}`: the `3xN` sorted positions.
- `original_index::Vector{Int32}`: the sorted-slot → original-particle permutation (the
  inverse step-2 gather).
- `σ::T`, `Σ::T`: the physical and modified-kernel widths.
- `h::T`, `inv_h::T`: the grid spacing and its inverse.
- `num_grid_points::NTuple{3, Int32}`: the grid dimensions `(M_x, M_y, M_z)`.
- `M_G::Int32`: the cubic stencil support per axis.
- `stencil_gaussian`, `stencil_r²`, `stencil_index`: the per-particle stencil scratch
  fields of length `M_G` (`StructArray`s of per-axis Gaussian weights, axis-squared
  distances, and periodic-wrapped 1-based indices).

# Returns
- `V`: the same matrix, holding the interpolated velocities.

# Notes
Preconditions (caller-guaranteed, so the loops are `@inbounds`): `V`, `Y_sorted` have shape
`(3, N)`; the stencil scratch fields have length `M_G`; `fluid_velocity` is backed by three
`Array{T, 3}` of shape `(M_x, M_y, M_z)` via `StructArrays.components`; `original_index` is
a permutation of `1:N`; positions have been folded into `[0, L_i)` by `wrap_positions!`.

See `spec/interpolation.md`.
"""
function _interpolate_velocities_kernel!(
    V::AbstractMatrix{T},
    fluid_velocity,
    Y_sorted::AbstractMatrix{T},
    original_index::Vector{Int32},
    σ::T,
    Σ::T,
    h::T,
    inv_h::T,
    num_grid_points::NTuple{3, Int32},
    M_G::Int32,
    stencil_gaussian,
    stencil_r²,
    stencil_index,
) where {T}
    gaussian_x, gaussian_y, gaussian_z = components(stencil_gaussian)
    r²_x, r²_y, r²_z = components(stencil_r²)
    idx_x, idx_y, idx_z = components(stencil_index)

    a₀, a₂, inv_norm, inv_2Σ² = _modified_kernel_coefficients(σ, Σ)
    h³ = h^3

    half_M_G = M_G ÷ Int32(2)

    @inbounds for s in axes(Y_sorted, 2)
        _fill_particle_stencil!(
            stencil_gaussian, stencil_r², stencil_index,
            Y_sorted[1, s], Y_sorted[2, s], Y_sorted[3, s],
            inv_norm, inv_2Σ², h, inv_h, num_grid_points, M_G, half_M_G,
        )

        # The SVector accumulator stays register-resident and allocation-free
        # (benchmarked at parity with three scalar accumulators).
        v = zero(SVector{3, T})
        for kz in Int32(1):M_G
            iz = idx_z[kz]
            gz = gaussian_z[kz]
            r²z = r²_z[kz]
            for ky in Int32(1):M_G
                iy = idx_y[ky]
                gaussian_yz = gaussian_y[ky] * gz
                r²_yz = r²_y[ky] + r²z
                for kx in Int32(1):M_G
                    ix = idx_x[kx]
                    w = (a₀ + a₂ * (r²_x[kx] + r²_yz)) * gaussian_x[kx] * gaussian_yz
                    v += w * fluid_velocity[ix, iy, iz]
                end
            end
        end

        n = original_index[s]
        V[:, n] .= h³ * v
    end
    return V
end
