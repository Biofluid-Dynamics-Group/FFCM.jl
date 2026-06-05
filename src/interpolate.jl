"""
    interpolate_velocities!(V, config) -> V

Step 5 of the Fast FCM algorithm (Su & Keaveny 2024, §4 Step 5). Interpolate
the fluid velocity field `config.fluid_velocity` (output of `stokes_solve!`)
to each particle position, evaluating the modified-kernel volume average
`Ṽ_n = ∫ u(x) Δ̃_n(x; Σ) d³x` (paper eq 283) by the trapezoidal rule over the
same `M_G³` stencil as the spread, with weight `h³`. Reads `config.Y_sorted`
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
        config.fluid_velocity,
        config.Y_sorted,
        config.original_index,
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
    return V
end

"""
    _interpolate_velocities_kernel!(
        V, fluid_velocity, Y_sorted, original_index,
        σ, Σ, h, inv_h, num_grid_points, M_G,
        gaussian_x, gaussian_y, gaussian_z,
        r²_x, r²_y, r²_z,
        idx_x, idx_y, idx_z,
    ) -> V

Function-barrier kernel for `interpolate_velocities!`. Per particle: anchor
the stencil at `j_i = round(Y_{n,i}/h)` and precompute the per-axis 1-D
Gaussian weights, axis-squared distances, and periodic-wrapped 1-based
stencil indices — identical to `_spread_forces_kernel!`. Then gather
`u(x_g) · (a₀ + a₂·r²) · g_x·g_y·g_z` over the `M_G³` stencil into scalar
accumulators, scale the result by `h³`, and write it to
`V[:, original_index[s]]` (the inverse step-2 permutation). The polynomial
coefficients `a₀ = 1 − 3·σ²_minus_Σ²/(2Σ²)`, `a₂ = σ²_minus_Σ²/(2Σ⁴)` with
`σ²_minus_Σ² = σ² − Σ²` follow the closed-form expansion of paper eq 267.

Preconditions (caller-guaranteed, so the loops are `@inbounds`): `V`,
`Y_sorted` have shape `(3, N)`; the scratch vectors have length `M_G`;
`fluid_velocity` is backed by three `Array{T, 3}` of shape `(M_x, M_y, M_z)`
via `StructArrays.components`; `original_index` is a permutation of `1:N`;
positions have been folded into `[0, L_i)` by `wrap_positions!`.

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
    ux, uy, uz = components(fluid_velocity)

    a_0, a_2, inv_norm, inv_2Σ² = _modified_kernel_coefficients(σ, Σ)
    h³ = h * h * h

    half_M_G = M_G ÷ Int32(2)

    @inbounds for s in axes(Y_sorted, 2)
        _fill_particle_stencil!(
            gaussian_x, gaussian_y, gaussian_z,
            r²_x, r²_y, r²_z,
            idx_x, idx_y, idx_z,
            Y_sorted[1, s], Y_sorted[2, s], Y_sorted[3, s],
            inv_norm, inv_2Σ², h, inv_h, num_grid_points, M_G, half_M_G,
        )

        vx = zero(T)
        vy = zero(T)
        vz = zero(T)
        @inbounds for kz in Int32(1):M_G
            iz = idx_z[kz]
            gz = gaussian_z[kz]
            r²z = r²_z[kz]
            for ky in Int32(1):M_G
                iy = idx_y[ky]
                gaussian_yz = gaussian_y[ky] * gz
                r²_yz = r²_y[ky] + r²z
                for kx in Int32(1):M_G
                    ix = idx_x[kx]
                    w = (a_0 + a_2 * (r²_x[kx] + r²_yz)) * gaussian_x[kx] * gaussian_yz
                    vx += ux[ix, iy, iz] * w
                    vy += uy[ix, iy, iz] * w
                    vz += uz[ix, iy, iz] * w
                end
            end
        end

        n = original_index[s]
        V[1, n] = h³ * vx
        V[2, n] = h³ * vy
        V[3, n] = h³ * vz
    end
    return V
end
