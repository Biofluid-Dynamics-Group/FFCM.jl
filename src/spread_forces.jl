"""
    spread_forces!(config) -> config

Step 3 of the Fast FCM algorithm (Su & Keaveny 2024, §4 Step 3). Evaluate
`J̃†[F](x_g) = Σₙ Fₙ Δ̃ₙ(x_g; Σ)` on every grid point `x_g`, writing the
result into `config.force_grid` (zeroed at the start of the call). Reads
`config.Y_sorted` and `config.F_sorted` (populated by
`sort_particles_by_cell!`). Allocation-free and type-stable on `T`.

See `spec/force-spreading.md`.
"""
function spread_forces!(config::FFCMConfig{T}) where {T}
    _spread_forces_kernel!(
        config.force_grid,
        config.Y_sorted,
        config.F_sorted,
        config.σ,
        config.Σ,
        config.Δx,
        config.inv_Δx,
        config.num_grid_points,
        config.M_G,
        config.gauss_x,
        config.gauss_y,
        config.gauss_z,
        config.r²_x,
        config.r²_y,
        config.r²_z,
        config.ind_x,
        config.ind_y,
        config.ind_z,
    )
    return config
end

"""
    _spread_forces_kernel!(
        force_grid, Y_sorted, F_sorted,
        σ, Σ, Δx, inv_Δx, num_grid_points, M_G,
        gauss_x, gauss_y, gauss_z,
        r²_x, r²_y, r²_z,
        ind_x, ind_y, ind_z,
    ) -> force_grid

Function-barrier kernel for `spread_forces!`. Per particle: anchor the
stencil at `j_i = round(Y_{n,i}/Δx)` (cuFCM convention, paper Table 2
calibration), precompute the per-axis 1-D Gaussian weights, axis-squared
distances, and periodic-wrapped 1-based stencil indices, then accumulate
`F_n · (a₀ + a₂·r²) · g_x·g_y·g_z` into the SoA components of
`force_grid`. The polynomial coefficients
`a₀ = 1 - 3·pdmag/(2Σ²)`, `a₂ = pdmag/(2Σ⁴)` with `pdmag = σ² - Σ²` follow
the closed-form expansion of paper eq 267.

Preconditions (caller-guaranteed, so the loops are `@inbounds`):
`Y_sorted`, `F_sorted` have shape `(3, N)`; the scratch vectors have
length `M_G`; `force_grid` is backed by three `Array{T, 3}` of shape
`(M_x, M_y, M_z)` via `StructArrays.components`; positions have been
folded into `[0, L_i)` by `wrap_positions!`.

See `spec/force-spreading.md`.
"""
function _spread_forces_kernel!(
    force_grid,
    Y_sorted::AbstractMatrix{T},
    F_sorted::AbstractMatrix{T},
    σ::T,
    Σ::T,
    Δx::T,
    inv_Δx::T,
    num_grid_points::NTuple{3, Int32},
    M_G::Int32,
    gauss_x::Vector{T},
    gauss_y::Vector{T},
    gauss_z::Vector{T},
    r²_x::Vector{T},
    r²_y::Vector{T},
    r²_z::Vector{T},
    ind_x::Vector{Int32},
    ind_y::Vector{Int32},
    ind_z::Vector{Int32},
) where {T}
    fx, fy, fz = components(force_grid)
    fill!(fx, zero(T))
    fill!(fy, zero(T))
    fill!(fz, zero(T))

    Σ² = Σ * Σ
    Σ⁴ = Σ² * Σ²
    pdmag = σ * σ - Σ²
    a_0 = one(T) - T(3) * pdmag / (T(2) * Σ²)
    a_2 = pdmag / (T(2) * Σ⁴)
    inv_norm = one(T) / sqrt(T(2) * T(π) * Σ²)
    inv_2Σ² = one(T) / (T(2) * Σ²)

    ngdh = M_G ÷ Int32(2)

    @inbounds for s in axes(Y_sorted, 2)
        F1 = F_sorted[1, s]
        F2 = F_sorted[2, s]
        F3 = F_sorted[3, s]

        _fill_particle_stencil!(
            gauss_x, gauss_y, gauss_z,
            r²_x, r²_y, r²_z,
            ind_x, ind_y, ind_z,
            Y_sorted[1, s], Y_sorted[2, s], Y_sorted[3, s],
            inv_norm, inv_2Σ², Δx, inv_Δx, num_grid_points, M_G, ngdh,
        )

        @inbounds for kz in Int32(1):M_G
            iz = ind_z[kz]
            gz = gauss_z[kz]
            r²z = r²_z[kz]
            for ky in Int32(1):M_G
                iy = ind_y[ky]
                gauss_yz = gauss_y[ky] * gz
                r²_yz = r²_y[ky] + r²z
                @simd for kx in Int32(1):M_G
                    ix = ind_x[kx]
                    w = (a_0 + a_2 * (r²_x[kx] + r²_yz)) * gauss_x[kx] * gauss_yz
                    fx[ix, iy, iz] += F1 * w
                    fy[ix, iy, iz] += F2 * w
                    fz[ix, iy, iz] += F3 * w
                end
            end
        end
    end
    return force_grid
end
