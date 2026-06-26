"""
    stokes_solve!(config) -> config

Step 4 of the Fast FCM algorithm (Su & Keaveny 2024, §4; the Fourier-space Stokes inversion
of §3, equations (32)-(33)). Apply the inverse Stokes operator `L^{-1}` to the spread force
field in `config.grid.force_density` and write the resulting fluid velocity field to
`config.grid.fluid_velocity`: a forward FFT of each `force_density` component into `fluid_hat`,
the per-mode projection `(I - k̂⊗k̂)/(μ k²) / M` in place on `fluid_hat` with the `k = 0`
mode zeroed (the mean-flow gauge fix), and an inverse FFT into `fluid_velocity`. The `1/M`
factor compensates the unnormalised FFTW round-trip.

# Arguments
- `config::FFCMConfig{T}`: the compiled configuration. Reads
  `config.grid.force_density` (populated by `spread_forces!`); writes
  `config.grid.fluid_velocity` and overwrites `config.solver.fluid_hat`.

# Returns
- `config`: the same configuration, with `config.grid.fluid_velocity` holding the Stokes velocity
  field.

See `spec/stokes-solve.md`.
"""
function stokes_solve!(config::FFCMConfig{T}) where {T}
    fx, fy, fz = components(config.grid.force_density)
    fx̂, fŷ, fẑ = components(config.solver.fluid_hat)
    mul!(fx̂, config.solver.forward_fourier_transform, fx)
    mul!(fŷ, config.solver.forward_fourier_transform, fy)
    mul!(fẑ, config.solver.forward_fourier_transform, fz)

    M_x, M_y, M_z = config.num_grid_points
    M = Int(M_x) * Int(M_y) * Int(M_z)
    inv_M = one(T) / T(M)
    _apply_inverse_stokes_kernel!(
        fx̂, fŷ, fẑ,
        config.solver.k_x, config.solver.k_y, config.solver.k_z,
        config.μ, inv_M,
    )

    ux, uy, uz = components(config.grid.fluid_velocity)
    mul!(ux, config.solver.inverse_fourier_transform, fx̂)
    mul!(uy, config.solver.inverse_fourier_transform, fŷ)
    mul!(uz, config.solver.inverse_fourier_transform, fẑ)
    return config
end

"""
    _apply_inverse_stokes_kernel!(
        fx̂, fŷ, fẑ, k_x, k_y, k_z, μ, inv_M,
    ) -> nothing

Kernel for `stokes_solve!`: the in-place Fourier-space projection. For each `(ix, iy, iz)`
with `k = (k_x[ix], k_y[iy], k_z[iz])`, compute `k² = kᵀk`, `α = inv_M / (μ k²)`,
`c = (k ⋅ f̂) / k²`, and replace `f̂ ← α ⋅ (f̂ - k ⋅ c)` component-wise. The `k = 0` mode is
fixed to zero.

# Arguments
- `fx̂`, `fŷ`, `fẑ`: the three `Array{Complex{T}, 3}` Fourier-space force components;
  overwritten in place with the projected velocity.
- `k_x::Vector{T}`, `k_y::Vector{T}`, `k_z::Vector{T}`: the per-axis wavevector components.
- `μ::T`: the fluid viscosity.
- `inv_M::T`: the `1/M` FFTW round-trip normalisation, `M = M_x⋅M_y⋅M_z`.

# Returns
- `nothing`. The three component arrays are overwritten in place.

# Notes
Preconditions (caller-guaranteed, so the loops are `@inbounds`): the three component arrays
are `Array{Complex{T}, 3}` of shape `(length(k_x), length(k_y), length(k_z))`;
`k_x[1] = k_y[1] = k_z[1] = 0` (the FFTW wrap-around layout puts the zero mode at the
leading index).

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
    zero_mode = zero(SVector{3, Complex{T}})
    @inbounds for iz in eachindex(k_z)
        kz = k_z[iz]
        for iy in eachindex(k_y)
            ky = k_y[iy]
            for ix in eachindex(k_x)
                k = SVector(k_x[ix], ky, kz)
                k² = dot(k, k)
                if iszero(k²)
                    # Gauge-fix the k = 0 mode (periodic Stokes is undefined there).
                    f̂ = zero_mode
                else
                    f̂ = SVector(fx̂[ix, iy, iz], fŷ[ix, iy, iz], fẑ[ix, iy, iz])
                    inv_k² = one(T) / k²
                    # Project the force transverse to k and scale by the Stokeslet
                    # 1/(μ k²): f̂ ← (1/(μ k²)) (I − k̂⊗k̂) f̂, with the unnormalised
                    # FFTW round-trip's 1/M folded into inv_μ_M. `dot(k, k)` and
                    # `dot(k, f̂)` are unrolled over the three axes (k is real, so
                    # `dot` adds no conjugation), keeping the projection in the
                    # vector form of paper §3 equations (32)–(33).
                    f̂ = (inv_μ_M * inv_k²) * (f̂ - k * (dot(k, f̂) * inv_k²))
                end
                fx̂[ix, iy, iz] = f̂[1]
                fŷ[ix, iy, iz] = f̂[2]
                fẑ[ix, iy, iz] = f̂[3]
            end
        end
    end
    return nothing
end
