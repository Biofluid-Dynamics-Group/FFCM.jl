"""
    interpolate_velocities!(V, config) -> V

Step 5 of the Fast FCM algorithm (Su & Keaveny 2024, §4 Step 5). Interpolate
the fluid velocity field `config.velocity_grid` (output of `stokes_solve!`)
to each particle position, evaluating the modified-kernel volume average
`Ṽ_n = ∫ u(x) Δ̃_n(x; Σ) d³x` (paper eq 283) by the trapezoidal rule over the
same `M_G³` stencil as the spread, with weight `Δx³`. Reads `config.Y_sorted`
and `config.original_index` (populated by `sort_particles_by_cell!`).

Writes the particle velocities into `V`, a caller-owned `3×N` matrix in the
caller's **original** particle order: the scatter-back via `original_index`
is the inverse of step 2's gather, which makes interpolation the exact
discrete adjoint of `spread_forces!` and the assembled mobility operator
symmetric positive-definite.

Allocation-free and type-stable on `T <: AbstractFloat`.

See `spec/interpolation.md`.
"""
function interpolate_velocities!(
    V::AbstractMatrix{T}, config::FFCMConfig{T},
) where {T}
    _interpolate_velocities_kernel!(
        V,
        config.velocity_grid,
        config.Y_sorted,
        config.original_index,
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
    return V
end

"""
    _interpolate_velocities_kernel!(
        V, velocity_grid, Y_sorted, original_index,
        σ, Σ, Δx, inv_Δx, num_grid_points, M_G,
        gauss_x, gauss_y, gauss_z,
        r²_x, r²_y, r²_z,
        ind_x, ind_y, ind_z,
    ) -> V

Function-barrier kernel for `interpolate_velocities!`. Per particle: anchor
the stencil at `j_i = round(Y_{n,i}/Δx)` and precompute the per-axis 1-D
Gaussian weights, axis-squared distances, and periodic-wrapped 1-based
stencil indices — identical to `_spread_forces_kernel!`. Then gather
`u(x_g) · (a₀ + a₂·r²) · g_x·g_y·g_z` over the `M_G³` stencil into scalar
accumulators, scale the result by `Δx³`, and write it to
`V[:, original_index[s]]` (the inverse step-2 permutation). The polynomial
coefficients `a₀ = 1 − 3·pdmag/(2Σ²)`, `a₂ = pdmag/(2Σ⁴)` with
`pdmag = σ² − Σ²` follow the closed-form expansion of paper eq 267.

Preconditions (caller-guaranteed, so the loops are `@inbounds`): `V`,
`Y_sorted` have shape `(3, N)`; the scratch vectors have length `M_G`;
`velocity_grid` is backed by three `Array{T, 3}` of shape `(M_x, M_y, M_z)`
via `StructArrays.components`; `original_index` is a permutation of `1:N`;
positions have been folded into `[0, L_i)` by `wrap_positions!`.

See `spec/interpolation.md`.
"""
function _interpolate_velocities_kernel!(
    V::AbstractMatrix{T},
    velocity_grid,
    Y_sorted::AbstractMatrix{T},
    original_index::Vector{Int32},
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
    ux, uy, uz = components(velocity_grid)

    Σ² = Σ * Σ
    Σ⁴ = Σ² * Σ²
    pdmag = σ * σ - Σ²
    a_0 = one(T) - T(3) * pdmag / (T(2) * Σ²)
    a_2 = pdmag / (T(2) * Σ⁴)
    inv_norm = one(T) / sqrt(T(2) * T(π) * Σ²)
    inv_2Σ² = one(T) / (T(2) * Σ²)
    Δx³ = Δx * Δx * Δx

    ngdh = M_G ÷ Int32(2)

    @inbounds for s in axes(Y_sorted, 2)
        _fill_particle_stencil!(
            gauss_x, gauss_y, gauss_z,
            r²_x, r²_y, r²_z,
            ind_x, ind_y, ind_z,
            Y_sorted[1, s], Y_sorted[2, s], Y_sorted[3, s],
            inv_norm, inv_2Σ², Δx, inv_Δx, num_grid_points, M_G, ngdh,
        )

        vx = zero(T)
        vy = zero(T)
        vz = zero(T)
        @inbounds for kz in Int32(1):M_G
            iz = ind_z[kz]
            gz = gauss_z[kz]
            r²z = r²_z[kz]
            for ky in Int32(1):M_G
                iy = ind_y[ky]
                gauss_yz = gauss_y[ky] * gz
                r²_yz = r²_y[ky] + r²z
                for kx in Int32(1):M_G
                    ix = ind_x[kx]
                    w = (a_0 + a_2 * (r²_x[kx] + r²_yz)) * gauss_x[kx] * gauss_yz
                    vx += ux[ix, iy, iz] * w
                    vy += uy[ix, iy, iz] * w
                    vz += uz[ix, iy, iz] * w
                end
            end
        end

        n = original_index[s]
        V[1, n] = Δx³ * vx
        V[2, n] = Δx³ * vy
        V[3, n] = Δx³ * vz
    end
    return V
end
