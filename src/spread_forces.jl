"""
    spread_forces!(config) -> config

Step 3 of the Fast FCM algorithm (Su & Keaveny 2024, §4; the spreading operator
of §3 equation (25)). Evaluate the spread force density `J̃†[F](x_g) = Σₙ Fₙ Δ̃ₙ(x_g; Σ)` on
every grid point `x_g`, writing the result into `config.force_density` (zeroed at the start
of the call).

# Arguments
- `config::FFCMConfig{T}`: the compiled configuration. Reads `config.Y_sorted` and
  `config.F_sorted` (populated by `sort_particles_by_cell!`).

# Returns
- `config`: the same configuration, with `config.force_density` overwritten by the spread
  force density.

See `spec/force-spreading.md`.
"""
function spread_forces!(config::FFCMConfig{T}) where {T}
    _spread_forces_kernel!(
        config.force_density,
        config.Y_sorted,
        config.F_sorted,
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
    return config
end

"""
    _spread_forces_kernel!(
        force_density, Y_sorted, F_sorted,
        σ, Σ, h, inv_h, num_grid_points, M_G,
        stencil_gaussian, stencil_r², stencil_index,
    ) -> force_density

Kernel for `spread_forces!`. For each particle, anchor the stencil at
`j_i = round(Y_{n,i}/h)` (cuFCM convention, paper Table 1 calibration), fill the per-axis
1-D Gaussian weights, axis-squared distances, and periodic-wrapped 1-based stencil indices
via `_fill_particle_stencil!`, and accumulate `F_n ⋅ (a₀ + a₂⋅r²) ⋅ g_x⋅g_y⋅g_z` into the
SoA components of `force_density`. The polynomial coefficients `(a₀, a₂)` and the Gaussian
normalisation come from `_modified_kernel_coefficients` (paper §3 equation (22)).

# Arguments
- `force_density`: the `StructArray{SVector{3, T}}` grid field to overwrite; zeroed at
  entry.
- `Y_sorted::AbstractMatrix{T}`, `F_sorted::AbstractMatrix{T}`: the `3xN` sorted positions
  and forces.
- `σ::T`, `Σ::T`: the original and modified-kernel widths.
- `h::T`, `inv_h::T`: the grid spacing and its inverse.
- `num_grid_points::NTuple{3, Int32}`: the grid dimensions `(M_x, M_y, M_z)`.
- `M_G::Int32`: the cubic stencil support per axis.
- `stencil_gaussian`, `stencil_r²`, `stencil_index`: the per-particle stencil scratch
  fields of length `M_G` (`StructArray`s of per-axis Gaussian weights, axis-squared
  distances, and periodic-wrapped 1-based indices).

# Returns
- `force_density`: the same grid field, holding the spread force density.

# Notes
Preconditions (caller-guaranteed, so the loops are `@inbounds`): `Y_sorted`, `F_sorted` have
shape `(3, N)`; the stencil scratch fields have length `M_G`; `force_density` and the
scratch fields are backed by per-component arrays via `StructArrays.components`
(`force_density` by three `Array{T, 3}` of shape `(M_x, M_y, M_z)`); positions have been
folded into `[0, L_i)` by `wrap_positions!`.

See `spec/force-spreading.md`.
"""
function _spread_forces_kernel!(
    force_density,
    Y_sorted::AbstractMatrix{T},
    F_sorted::AbstractMatrix{T},
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
    fx, fy, fz = components(force_density)
    fill!(fx, zero(T))
    fill!(fy, zero(T))
    fill!(fz, zero(T))

    gaussian_x, gaussian_y, gaussian_z = components(stencil_gaussian)
    r²_x, r²_y, r²_z = components(stencil_r²)
    idx_x, idx_y, idx_z = components(stencil_index)

    a₀, a₂, inv_norm, inv_2Σ² = _modified_kernel_coefficients(σ, Σ)

    half_M_G = M_G ÷ Int32(2)

    @inbounds for s in axes(Y_sorted, 2)
        F_n = SVector(F_sorted[1, s], F_sorted[2, s], F_sorted[3, s])

        _fill_particle_stencil!(
            stencil_gaussian, stencil_r², stencil_index,
            Y_sorted[1, s], Y_sorted[2, s], Y_sorted[3, s],
            inv_norm, inv_2Σ², h, inv_h, num_grid_points, M_G, half_M_G,
        )

        for kz in Int32(1):M_G
            iz = idx_z[kz]
            gz = gaussian_z[kz]
            r²z = r²_z[kz]
            for ky in Int32(1):M_G
                iy = idx_y[ky]
                gaussian_yz = gaussian_y[ky] * gz
                r²_yz = r²_y[ky] + r²z
                # The SVector store through the StructArray lowers to the
                # same three SoA component writes as explicit fx/fy/fz
                # accumulation (benchmarked at parity, allocation-free), and
                # `@simd` vectorisation of the kx sweep survives it.
                @simd for kx in Int32(1):M_G
                    ix = idx_x[kx]
                    w = (a₀ + a₂ * (r²_x[kx] + r²_yz)) * gaussian_x[kx] * gaussian_yz
                    force_density[ix, iy, iz] += w * F_n
                end
            end
        end
    end
    return force_density
end
