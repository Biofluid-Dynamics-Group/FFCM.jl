"""
    spread_forces!(config) -> config

Step 3 of the Fast FCM algorithm (Su & Keaveny 2024, §4; the spreading operator
of §3 equation (25)). Evaluate the spread force density
`J̃†[F](x_g) = Σₙ Fₙ Δ̃ₙ(x_g; Σ)` on every grid point `x_g`, writing the result
into `config.force_density` (zeroed at the start of the call). Allocation-free
and type-stable on `T`.

# Arguments
- `config::FFCMConfig{T}`: the compiled configuration. Reads `config.Y_sorted`
  and `config.F_sorted` (populated by `sort_particles_by_cell!`).

# Returns
- `config`: the same configuration, with `config.force_density` overwritten by
  the spread force density.

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
        config.gaussian_x,
        config.gaussian_y,
        config.gaussian_z,
        config.r²_x,
        config.r²_y,
        config.r²_z,
        config.idx_x,
        config.idx_y,
        config.idx_z,
    )
    return config
end

"""
    _spread_forces_kernel!(
        force_density, Y_sorted, F_sorted,
        σ, Σ, h, inv_h, num_grid_points, M_G,
        gaussian_x, gaussian_y, gaussian_z,
        r²_x, r²_y, r²_z,
        idx_x, idx_y, idx_z,
    ) -> force_density

Function-barrier kernel for `spread_forces!`. Per particle: anchor the stencil
at `j_i = round(Y_{n,i}/h)` (cuFCM convention, paper Table 1 calibration),
precompute the per-axis 1-D Gaussian weights, axis-squared distances, and
periodic-wrapped 1-based stencil indices (via `_fill_particle_stencil!`), then
accumulate `F_n · (a₀ + a₂·r²) · g_x·g_y·g_z` into the SoA components of
`force_density`. The polynomial coefficients `(a₀, a₂)` and the Gaussian
normalisation come from `_modified_kernel_coefficients` — the closed-form
expansion of paper §3 equation (22).

# Arguments
- `force_density`: the `StructArray{SVector{3, T}}` grid field to overwrite;
  zeroed at entry.
- `Y_sorted::AbstractMatrix{T}`, `F_sorted::AbstractMatrix{T}`: the `3×N` sorted
  positions and forces.
- `σ::T`, `Σ::T`: the physical and modified-kernel widths.
- `h::T`, `inv_h::T`: the grid spacing and its inverse.
- `num_grid_points::NTuple{3, Int32}`: the grid dimensions `(M_x, M_y, M_z)`.
- `M_G::Int32`: the cubic stencil support per axis.
- `gaussian_x`, `gaussian_y`, `gaussian_z`, `r²_x`, `r²_y`, `r²_z`, `idx_x`,
  `idx_y`, `idx_z`: per-axis scratch vectors of length `M_G` (Gaussian weights,
  axis-squared distances, and periodic-wrapped 1-based indices).

# Returns
- `force_density`: the same grid field, holding the spread force density.

# Notes
Preconditions (caller-guaranteed, so the loops are `@inbounds`): `Y_sorted`,
`F_sorted` have shape `(3, N)`; the scratch vectors have length `M_G`;
`force_density` is backed by three `Array{T, 3}` of shape `(M_x, M_y, M_z)` via
`StructArrays.components`; positions have been folded into `[0, L_i)` by
`wrap_positions!`.

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
    gaussian_x::Vector{T},
    gaussian_y::Vector{T},
    gaussian_z::Vector{T},
    r²_x::Vector{T},
    r²_y::Vector{T},
    r²_z::Vector{T},
    idx_x::Vector{Int32},
    idx_y::Vector{Int32},
    idx_z::Vector{Int32},
) where {T}
    fx, fy, fz = components(force_density)
    fill!(fx, zero(T))
    fill!(fy, zero(T))
    fill!(fz, zero(T))

    a_0, a_2, inv_norm, inv_2Σ² = _modified_kernel_coefficients(σ, Σ)

    half_M_G = M_G ÷ Int32(2)

    @inbounds for s in axes(Y_sorted, 2)
        F1 = F_sorted[1, s]
        F2 = F_sorted[2, s]
        F3 = F_sorted[3, s]

        _fill_particle_stencil!(
            gaussian_x, gaussian_y, gaussian_z,
            r²_x, r²_y, r²_z,
            idx_x, idx_y, idx_z,
            Y_sorted[1, s], Y_sorted[2, s], Y_sorted[3, s],
            inv_norm, inv_2Σ², h, inv_h, num_grid_points, M_G, half_M_G,
        )

        @inbounds for kz in Int32(1):M_G
            iz = idx_z[kz]
            gz = gaussian_z[kz]
            r²z = r²_z[kz]
            for ky in Int32(1):M_G
                iy = idx_y[ky]
                gaussian_yz = gaussian_y[ky] * gz
                r²_yz = r²_y[ky] + r²z
                # The three force components are accumulated into separate SoA
                # arrays (fx/fy/fz), not an SVector per grid point: this lets
                # `@simd` vectorise the inner kx sweep and keeps the kernel
                # allocation-free. A more vectorial form
                # is deferred: the risk is the `@simd` independence, so it is
                # benchmark-gated and must keep
                # `@ballocated == 0`/`@inferred` green before it replaces this.
                @simd for kx in Int32(1):M_G
                    ix = idx_x[kx]
                    w = (a_0 + a_2 * (r²_x[kx] + r²_yz)) * gaussian_x[kx] * gaussian_yz
                    fx[ix, iy, iz] += F1 * w
                    fy[ix, iy, iz] += F2 * w
                    fz[ix, iy, iz] += F3 * w
                end
            end
        end
    end
    return force_density
end
