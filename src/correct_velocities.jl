"""
    _self_correction(σ, Σ, a, μ) -> T

The `r → 0` limit of the pairwise velocity correction (paper Appendix B, equation (B.1)): a
scalar, the diagonal self-mobility correction `M^VF_nn - M̃^VF_nn` applied to every particle
independent of its neighbours. With `a = σ√π` (paper §2 radius-width relation) and `η = μ`,

    self_correction_term = 1/(6πμa) - 1/(6πμ⋅Σ√π) + (σ²−Σ²)/(12μ(Σ√π)³) - (σ²−Σ²)²/(32μ⋅Σ⁵⋅π^{3/2}).

It vanishes at `Σ = σ` (the standard-FCM limit).

# Arguments
- `σ::T`: the physical kernel width.
- `Σ::T`: the modified-kernel width.
- `a::T`: the particle radius (`a = σ√π = 1` in the unit-radius convention).
- `μ::T`: the fluid viscosity.

# Returns
- `T`: the scalar self-correction term added to every particle's velocity.
"""
function _self_correction(σ::T, Σ::T, a::T, μ::T) where {T}
    σ²_minus_Σ² = σ^2 - Σ^2
    Σ_sqrtπ = Σ * sqrt(T(π))
    return one(T) / (T(6) * T(π) * μ * a) - one(T) / (T(6) * T(π) * μ * Σ_sqrtπ) +
           σ²_minus_Σ² / (T(12) * μ * Σ_sqrtπ^3) -
           σ²_minus_Σ²^2 / (T(32) * μ * Σ^5 * sqrt(T(π)^3))
end

"""
    _correction_scalars(r, σ, Σ, μ) -> Tuple{T, T}

The two pair scalars of the real-space velocity correction at separation `r`, collapsed from
the typo-corrected paper §3 equation (31): the correction tensor is
`isotropic_coefficient⋅I + parallel_coefficient⋅x⊗x`, so the velocity
of particle `n` from the force `F_m` on a neighbour `m` is

    ΔV_n += isotropic_coefficient⋅F_m + parallel_coefficient⋅(x⋅F_m)⋅x,  x = Y_n - Y_m.

With `η = μ`, `erf_{2w} = erf(r/(2w))`, and the √2-scaled-width Gaussians
`Δ(w) = (4πw²)^{-3/2}⋅e^{-r²/4w²}`, the two coefficients group by their transcendental
factor — the `erf` difference, `Δ(σ)`, and `Δ(Σ)` — as

    isotropic_coefficient =
        (erf_{2σ} − erf_{2Σ}) ⋅ (1/(8πμr) + σ²/(4πμr³))
        − (2σ⁴/(μr²)) ⋅ Δ(σ)
        + [2Σ⁴/(μr²) + ((σ²−Σ²)/μ)(1 + 2Σ²/r²)
           − ((σ²−Σ²)²/4) ⋅ (2 − r²/(2Σ²))/(2μΣ²)] ⋅ Δ(Σ)

    parallel_coefficient =
        (erf_{2σ} − erf_{2Σ}) ⋅ (1/(8πμr³) − 3σ²/(4πμr⁵))
        + (6σ⁴/(μr⁴)) ⋅ Δ(σ)
        + [−6Σ⁴/(μr⁴) − ((σ²−Σ²)/(μr²))(1 + 6Σ²/r²)
           − ((σ²−Σ²)²/4) ⋅ 1/(4μΣ⁴)] ⋅ Δ(Σ)

so only two `erf` and two `exp` are evaluated per interaction. The printed equation (31)
misprints the `erf` argument as `r/(w√2)`; the regularised Stokeslet of equations (12) and
(14) fixes it as `r/(2w)`, which this function implements.

# Arguments
- `r::T`: the centre-to-centre separation `|Y_n - Y_m|`. Must be `> 0`.
- `σ::T`, `Σ::T`: the physical and modified-kernel widths.
- `μ::T`: the fluid viscosity.

# Returns
- `Tuple{T, T}`: `(isotropic_coefficient, parallel_coefficient)` — the coefficient of `I`
  and the coefficient of `xxᵀ` (the component along the separation `x`).

# Notes
Valid for `r > 0`; the `r = 0` diagonal is the separate `_self_correction`.
"""
function _correction_scalars(r::T, σ::T, Σ::T, μ::T) where {T}
    r² = r^2
    r³ = r^3
    r⁴ = r²^2
    r⁵ = r⁴ * r
    σ² = σ^2
    σ⁴ = σ²^2
    Σ² = Σ^2
    Σ⁴ = Σ²^2
    σ²_minus_Σ² = σ² - Σ²
    σ²_minus_Σ²_sq_quarter = σ²_minus_Σ²^2 / T(4)
    fourπμ = T(4) * T(π) * μ
    eightπμ = T(8) * T(π) * μ

    erf_diff = erf(r / (T(2) * σ)) - erf(r / (T(2) * Σ))
    Δσ = (T(4) * T(π) * σ²)^(-T(3) / 2) * exp(-r² / (T(4) * σ²))
    ΔΣ = (T(4) * T(π) * Σ²)^(-T(3) / 2) * exp(-r² / (T(4) * Σ²))

    isotropic_coefficient =
        erf_diff * (one(T) / (eightπμ * r) + σ² / (fourπμ * r³)) -
        T(2) * σ⁴ / (μ * r²) * Δσ +
        (
            T(2) * Σ⁴ / (μ * r²) + σ²_minus_Σ² / μ * (one(T) + T(2) * Σ² / r²) -
            σ²_minus_Σ²_sq_quarter * (T(2) - r² / (T(2) * Σ²)) / (T(2) * μ * Σ²)
        ) * ΔΣ
    parallel_coefficient =
        erf_diff * (one(T) / (eightπμ * r³) - T(3) * σ² / (fourπμ * r⁵)) +
        T(6) * σ⁴ / (μ * r⁴) * Δσ +
        (
            -T(6) * Σ⁴ / (μ * r⁴) - σ²_minus_Σ² / (μ * r²) * (one(T) + T(6) * Σ² / r²) -
            σ²_minus_Σ²_sq_quarter / (T(4) * μ * Σ⁴)
        ) * ΔΣ
    return (isotropic_coefficient, parallel_coefficient)
end

"""
    _min_image(x, L) -> SVector{3, T}

Reduce the separation vector `x = Y_n - Y_m` to its nearest periodic image, folding each
axis to `[-L_i/2, L_i/2]` by `x_i - L_i⋅round(x_i/L_i)` (`RoundNearestTiesToEven`). Because
`round(-y) = -round(y)` for ties-to-even, the reduction is exactly antisymmetric
(`_min_image(-x) = -_min_image(x)`), which keeps the pair correction symmetric.

# Arguments
- `x::SVector{3, T}`: the raw separation vector `Y_n - Y_m`.
- `L::NTuple{3, T}`: the periodic box lengths.

# Returns
- `SVector{3, T}`: the minimum-image separation, each axis in `[-L_i/2, L_i/2]`.
"""
@inline function _min_image(x::SVector{3, T}, L::NTuple{3, T}) where {T}
    return SVector{3, T}(
        x[1] - L[1] * round(x[1] / L[1]),
        x[2] - L[2] * round(x[2] / L[2]),
        x[3] - L[3] * round(x[3] / L[3]),
    )
end

"""
    correct_velocities!(V, config) -> V

Step 6 of the Fast FCM algorithm (Su & Keaveny 2024, §4). Add the real-space pairwise
correction `M^VF - M̃^VF` (paper §3 equation (31)) and the per-particle self term (paper
Appendix B, equation (B.1)) to the interpolated velocities `V`, the same `3xN` buffer
`interpolate_velocities!` wrote, in the caller's **original** particle order. The correction
is **added** in place (`V[:, n] += ΔV_n`), realising `M = M̃ + (M - M̃)`.

Each particle gathers the correction from neighbours within `config.R_c` over the 27
surrounding cells, applying
`ΔV_n += isotropic_coefficient⋅F_m + parallel_coefficient⋅(x⋅F_m)⋅x` with the two pair
scalars of `_correction_scalars` and the closed-form self term. The result completes the
σ-regularised mobility, independent of Σ.

# Arguments
- `V::AbstractMatrix{T}`: the `3xN` velocity matrix to correct in place (the output of
  `interpolate_velocities!`, in original particle order).
- `config::FFCMConfig{T}`: the compiled configuration. Reads `config.particles.Y_sorted`,
  `config.particles.F_sorted`, the cell list (`config.cells.cell_start` / `config.cells.cell_end`), and
  `config.cells.original_index` (populated by `sort_particles_by_cell!`).

# Returns
- `V`: the same matrix, with the pairwise + self correction added.
"""
function correct_velocities!(V::AbstractMatrix{T}, config::FFCMConfig{T}) where {T}
    _correct_velocities_kernel!(
        V,
        config.particles.Y_sorted,
        config.particles.F_sorted,
        config.cells.cell_start,
        config.cells.cell_end,
        config.cells.original_index,
        config.cells.neighbor_map,
        config.num_cells,
        config.inv_cell_size,
        config.L,
        config.σ,
        config.Σ,
        config.a,
        config.μ,
        config.R_c,
    )
    return V
end

"""
    _correct_velocities_kernel!(
        V, Y_sorted, F_sorted, cell_start, cell_end, original_index, neighbor_map,
        num_cells, inv_cell_size, L, σ, Σ, a, μ, R_c,
    ) -> V

Kernel for `correct_velocities!`. Computes the self scalar once, then iterates cells in
increasing linear hash (`x` fastest, `z` slowest, matching `_assign_cells_kernel!`). For
each sorted particle `s` it seeds the self term `self_correction_term⋅F_s`, sweeps the 27
wrapped neighbour cells, and for every neighbour `s'` within `R_c` (minimum-image)
accumulates `isotropic_coefficient⋅F_{s'} + parallel_coefficient⋅(x⋅F_{s'})⋅x`. The result
is scattered to `V[:, original_index[s]]` via `+=` (inverse step-2 permutation). Each
particle writes only its own column — a gather, no write race.

# Arguments
- `V::AbstractMatrix{T}`: the `3xN` velocity matrix; corrected in place (`+=`).
- `Y_sorted::AbstractMatrix{T}`, `F_sorted::AbstractMatrix{T}`: the `3xN` sorted positions
  and forces.
- `cell_start::Vector{Int32}`, `cell_end::Vector{Int32}`: the 1-based inclusive per-cell
  slot ranges.
- `original_index::Vector{Int32}`: the sorted-slot to original-particle permutation.
- `neighbor_map::Nothing`: `nothing` on the CPU backend, which walks all 27 surrounding
  cells on the fly; the GPU backend passes the half-shell neighbour map here instead, and
  this argument is the dispatch discriminator between the two backends.
- `num_cells::NTuple{3, Int32}`: the cell-grid dimensions.
- `inv_cell_size::NTuple{3, T}`: the inverse cell sizes; unused by the CPU method (which
  walks cells directly), carried for the GPU backend that recomputes a particle's cell.
- `L::NTuple{3, T}`: the periodic box lengths.
- `σ::T`, `Σ::T`, `a::T`, `μ::T`: the kernel widths, particle radius, and viscosity.
- `R_c::T`: the correction cutoff radius.

# Returns
- `V`: the same matrix, with the correction scattered in.

# Notes
Preconditions (caller-guaranteed, so the loops are `@inbounds`): `V`, `Y_sorted`, `F_sorted`
have shape `(3, N)`; `cell_start`/`cell_end` have length `prod(num_cells)` with
`num_cells[i] ≥ 3`; `original_index` is a permutation of `1:N`; positions folded into
`[0, L_i)`; `R_c ≤ min(L)/2` (enforced by the constructor) so the minimum image is
unambiguous.
"""
function _correct_velocities_kernel!(
    V::AbstractMatrix{T},
    Y_sorted::AbstractMatrix{T},
    F_sorted::AbstractMatrix{T},
    cell_start::Vector{Int32},
    cell_end::Vector{Int32},
    original_index::Vector{Int32},
    neighbor_map::Nothing,
    num_cells::NTuple{3, Int32},
    inv_cell_size::NTuple{3, T},
    L::NTuple{3, T},
    σ::T,
    Σ::T,
    a::T,
    μ::T,
    R_c::T,
) where {T}
    self_correction_term = _self_correction(σ, Σ, a, μ)
    R_c² = R_c^2
    m_x, m_y, m_z = num_cells

    @inbounds for cz in Int32(0):(m_z - Int32(1))
        for cy in Int32(0):(m_y - Int32(1))
            for cx in Int32(0):(m_x - Int32(1))
                c = cx + (cy + cz * m_y) * m_x
                for s in cell_start[c + Int32(1)]:cell_end[c + Int32(1)]
                    Yn = SVector{3, T}(Y_sorted[1, s], Y_sorted[2, s], Y_sorted[3, s])
                    v = self_correction_term *
                        SVector{3, T}(F_sorted[1, s], F_sorted[2, s], F_sorted[3, s])
                    for dz in Int32(-1):Int32(1)
                        nz = mod(cz + dz, m_z)
                        for dy in Int32(-1):Int32(1)
                            ny = mod(cy + dy, m_y)
                            for dx in Int32(-1):Int32(1)
                                nx = mod(cx + dx, m_x)
                                cn = nx + (ny + nz * m_y) * m_x
                                for s2 in cell_start[cn + Int32(1)]:cell_end[cn + Int32(1)]
                                    s2 == s && continue
                                    Ym = SVector{3, T}(
                                        Y_sorted[1, s2], Y_sorted[2, s2], Y_sorted[3, s2],
                                    )
                                    x = _min_image(Yn - Ym, L)
                                    r² = dot(x, x)
                                    r² < R_c² || continue
                                    isotropic_coefficient, parallel_coefficient =
                                        _correction_scalars(sqrt(r²), σ, Σ, μ)
                                    Fm = SVector{3, T}(
                                        F_sorted[1, s2], F_sorted[2, s2], F_sorted[3, s2],
                                    )
                                    xdotF = dot(x, Fm)
                                    v +=
                                        isotropic_coefficient * Fm +
                                        (parallel_coefficient * xdotF) * x
                                end
                            end
                        end
                    end
                    n = original_index[s]
                    # Explicit per-component accumulation: the broadcast form
                    # `V[:, n] .+= v` routes the SubArray–SVector mix through
                    # StaticArrays' broadcast style and allocates a temporary,
                    # breaking the zero-allocation contract.
                    V[1, n] += v[1]
                    V[2, n] += v[2]
                    V[3, n] += v[3]
                end
            end
        end
    end
    return V
end
