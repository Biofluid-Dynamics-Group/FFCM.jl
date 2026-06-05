"""
    stokes_solve!(config) -> config

Step 4 of the Fast FCM algorithm (Su & Keaveny 2024, §4 Step 4 = §3
Step `solve`). Apply the inverse Stokes operator `L^{-1}` to the spread
force field in `config.force_density` and write the resulting fluid
velocity field to `config.fluid_velocity`. Reads `config.force_density`
(populated by `spread_forces!`); writes `config.fluid_velocity` and
overwrites `config.fluid_hat`.

Operationally:
  1. Forward r2c FFT each component of `force_density` into the
     corresponding component of `fluid_hat`.
  2. Apply `(I − k̂k̂ᵀ)/(μ k²) / M` per Fourier mode in place on
     `fluid_hat`; zero the `k = 0` mode (the mean-flow gauge fix).
  3. Backward c2r FFT each component of `fluid_hat` into the
     corresponding component of `fluid_velocity`.

The `1/M` factor compensates the unnormalised FFTW round-trip.

Allocation-free and type-stable on `T <: AbstractFloat`.

See `spec/stokes-solve.md`.
"""
function stokes_solve!(config::FFCMConfig{T}) where {T}
    fx, fy, fz = components(config.force_density)
    fx̂, fŷ, fẑ = components(config.fluid_hat)
    mul!(fx̂, config.forward_fourier_transform, fx)
    mul!(fŷ, config.forward_fourier_transform, fy)
    mul!(fẑ, config.forward_fourier_transform, fz)

    M_x, M_y, M_z = config.num_grid_points
    M = Int(M_x) * Int(M_y) * Int(M_z)
    inv_M = one(T) / T(M)
    _apply_inverse_stokes_kernel!(
        fx̂, fŷ, fẑ,
        config.k_x, config.k_y, config.k_z,
        config.μ, inv_M,
    )

    ux, uy, uz = components(config.fluid_velocity)
    mul!(ux, config.inverse_fourier_transform, fx̂)
    mul!(uy, config.inverse_fourier_transform, fŷ)
    mul!(uz, config.inverse_fourier_transform, fẑ)
    return config
end

"""
    _apply_inverse_stokes_kernel!(
        fx̂, fŷ, fẑ, k_x, k_y, k_z, μ, inv_M,
    ) -> nothing

Function-barrier kernel for `stokes_solve!`. In-place Fourier-space
projection: for each `(ix, iy, iz)` with
`k = (k_x[ix], k_y[iy], k_z[iz])`, compute `k² = kᵀk`,
`α = inv_M / (μ k²)`, `c = (k · f̂) / k²`, and replace
`f̂ ← α · (f̂ − k · c)` component-wise. The `k = 0` mode is gauge-fixed
to zero (paper §3; periodic Stokes is undefined there).

Preconditions (caller-guaranteed, so the loops are `@inbounds`):
the three component arrays are `Array{Complex{T}, 3}` of shape
`(length(k_x), length(k_y), length(k_z))`; `k_x[1] = k_y[1] = k_z[1] = 0`
(the FFTW wrap-around layout puts the zero mode at the leading index).

See `spec/stokes-solve.md`.
"""
function _apply_inverse_stokes_kernel!(
    fx̂::AbstractArray{Complex{T}, 3},
    fŷ::AbstractArray{Complex{T}, 3},
    fẑ::AbstractArray{Complex{T}, 3},
    k_x::Vector{T},
    k_y::Vector{T},
    k_z::Vector{T},
    μ::T,
    inv_M::T,
) where {T}
    inv_μ_M = inv_M / μ
    zero_C = zero(Complex{T})
    @inbounds for iz in eachindex(k_z)
        kz = k_z[iz]
        for iy in eachindex(k_y)
            ky = k_y[iy]
            for ix in eachindex(k_x)
                kx = k_x[ix]
                k² = kx * kx + ky * ky + kz * kz
                if iszero(k²)
                    fx̂[ix, iy, iz] = zero_C
                    fŷ[ix, iy, iz] = zero_C
                    fẑ[ix, iy, iz] = zero_C
                else
                    fx̂_v = fx̂[ix, iy, iz]
                    fŷ_v = fŷ[ix, iy, iz]
                    fẑ_v = fẑ[ix, iy, iz]
                    inv_k² = one(T) / k²
                    α = inv_μ_M * inv_k²
                    c = (kx * fx̂_v + ky * fŷ_v + kz * fẑ_v) * inv_k²
                    fx̂[ix, iy, iz] = α * (fx̂_v - kx * c)
                    fŷ[ix, iy, iz] = α * (fŷ_v - ky * c)
                    fẑ[ix, iy, iz] = α * (fẑ_v - kz * c)
                end
            end
        end
    end
    return nothing
end
