"""
    _self_correction(σ, Σ, a, μ) -> c_self

The `r → 0` limit of the pairwise velocity correction (paper
eq:correction_VF_limit, `appendix.tex:52-54`): a scalar, the diagonal
self-mobility correction `M^VF_nn − M̃^VF_nn` applied to every particle
independent of its neighbours. With `a = σ√π` (paper eq 163) and `η = μ`,

    c_self = 1/(6πμa) − 1/(6πμ·Σ√π) + (σ²−Σ²)/(12μ(Σ√π)³) − (σ²−Σ²)²/(32μ·Σ⁵·π^{3/2}).

Vanishes at `Σ = σ` (the standard-FCM degenerate limit). Pure and
allocation-free.

See `spec/pairwise-correction.md`.
"""
function _self_correction(σ::T, Σ::T, a::T, μ::T) where {T}
    σ² = σ * σ
    Σ² = Σ * Σ
    pd = σ² - Σ²
    Σπ = Σ * sqrt(T(π))
    return one(T) / (T(6) * T(π) * μ * a) - one(T) / (T(6) * T(π) * μ * Σπ) +
           pd / (T(12) * μ * Σπ^3) - pd * pd / (T(32) * μ * Σ²^2 * Σ * T(π)^(T(3) / 2))
end

"""
    _correction_scalars(r, σ, Σ, μ) -> (A, B)

The two pair scalars of the real-space velocity correction at separation `r`,
collapsed from the typo-corrected paper eq:correction_VF (`outline.tex:310-316`):
the correction tensor is `A·I + B·xxᵀ` (un-normalised `xxᵀ`), so the velocity of
particle `n` from the force `F_m` on a neighbour `m` is

    ΔV_n += A·F_m + B·(x·F_m)·x,    x = Y_n − Y_m.

`A` is the coefficient of `I`, `B` the coefficient of `xxᵀ`. With `η = μ`,
`pd = σ²−Σ²`, `erf_{2w} = erf(r/(2w))` (the √2-scaled-width argument; see the
spec for the line-312 typo note), and Gaussians `Δ_w = (4πw²)^{-3/2}·e^{-r²/4w²}`,
the grouped coefficients of `(erf_{2σ}−erf_{2Σ})`, `Δ_σ`, and `Δ_Σ` are as
documented in `spec/pairwise-correction.md`. Needs exactly two `erf` and two `exp`.

Pure and allocation-free. Valid for `0 < r`; the `r = 0` diagonal is the separate
`_self_correction`.

See `spec/pairwise-correction.md`.
"""
function _correction_scalars(r::T, σ::T, Σ::T, μ::T) where {T}
    r² = r * r
    r³ = r² * r
    r⁴ = r² * r²
    r⁵ = r⁴ * r
    σ² = σ * σ
    σ⁴ = σ² * σ²
    Σ² = Σ * Σ
    Σ⁴ = Σ² * Σ²
    pd = σ² - Σ²
    pd²q = pd * pd / T(4)
    fourπμ = T(4) * T(π) * μ
    eightπμ = T(8) * T(π) * μ

    erf_diff = erf(r / (T(2) * σ)) - erf(r / (T(2) * Σ))
    Δσ = (T(4) * T(π) * σ²)^(-T(3) / 2) * exp(-r² / (T(4) * σ²))
    ΔΣ = (T(4) * T(π) * Σ²)^(-T(3) / 2) * exp(-r² / (T(4) * Σ²))

    A = erf_diff * (one(T) / (eightπμ * r) + σ² / (fourπμ * r³)) -
        T(2) * σ⁴ / (μ * r²) * Δσ +
        (
            T(2) * Σ⁴ / (μ * r²) + pd / μ * (one(T) + T(2) * Σ² / r²) -
            pd²q * (T(2) - r² / (T(2) * Σ²)) / (T(2) * μ * Σ²)
        ) * ΔΣ
    B = erf_diff * (one(T) / (eightπμ * r³) - T(3) * σ² / (fourπμ * r⁵)) +
        T(6) * σ⁴ / (μ * r⁴) * Δσ +
        (
            -T(6) * Σ⁴ / (μ * r⁴) - pd / (μ * r²) * (one(T) + T(6) * Σ² / r²) -
            pd²q / (T(4) * μ * Σ⁴)
        ) * ΔΣ
    return (A, B)
end

"""
    _min_image(x, L) -> SVector{3}

Reduce the separation vector `x = Y_n − Y_m` to its nearest periodic image,
folding each axis to `[−L_i/2, L_i/2]` by `x_i − L_i·round(x_i/L_i)`
(`RoundNearestTiesToEven`). `round(−y) = −round(y)` for ties-to-even, so the
reduction is exactly antisymmetric (`_min_image(−x) = −_min_image(x)`), which
keeps the pair correction symmetric. Pure and allocation-free.
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

Step 6 of the Fast FCM algorithm (Su & Keaveny 2024, §4 Step 6). Add the
real-space pairwise correction `M^VF − M̃^VF` (paper eq:correction_VF) and the
per-particle self term (eq:correction_VF_limit) to the interpolated velocities
`V`, a caller-owned `3×N` matrix in the caller's **original** particle order — the
same buffer `interpolate_velocities!` wrote. The correction is **added** in place
(`V[:, n] += ΔV_n`), realising `M = M̃ + (M − M̃)`.

Reads `config.Y_sorted`, `config.F_sorted`, the cell list
(`config.cell_start` / `config.cell_end`), and `config.original_index`
(populated by `sort_particles_by_cell!`). Each particle gathers the correction
from neighbours within `config.R_c` over the 27 surrounding cells, applying
`ΔV_n += A(r)·F_m + B(r)·(x·F_m)·x` with the two pair scalars of
`_correction_scalars` and the closed-form self term. The result completes the
σ-regularised mobility, independent of Σ.

Allocation-free and type-stable on `T <: AbstractFloat`.

See `spec/pairwise-correction.md`.
"""
function correct_velocities!(V::AbstractMatrix{T}, config::FFCMConfig{T}) where {T}
    _correct_velocities_kernel!(
        V,
        config.Y_sorted,
        config.F_sorted,
        config.cell_start,
        config.cell_end,
        config.original_index,
        config.num_cells,
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
        V, Y_sorted, F_sorted, cell_start, cell_end, original_index,
        num_cells, L, σ, Σ, a, μ, R_c,
    ) -> V

Function-barrier kernel for `correct_velocities!`. Computes the self scalar once,
then iterates cells in increasing linear hash (`x` fastest, `z` slowest, matching
`_assign_cells_kernel!`). For each sorted particle `s` it seeds the self term
`c_self·F_s`, sweeps the 27 wrapped neighbour cells, and for every neighbour `s'`
within `R_c` (minimum-image) accumulates `A(r)·F_{s'} + B(r)·(x·F_{s'})·x`. The
result is scattered to `V[:, original_index[s]]` via `+=` (inverse step-2
permutation). Each particle writes only its own column — a gather, no write race.

Preconditions (caller-guaranteed, so the loops are `@inbounds`): `V`, `Y_sorted`,
`F_sorted` have shape `(3, N)`; `cell_start`/`cell_end` have length
`prod(num_cells)` with `num_cells[i] ≥ 3`; `original_index` is a permutation of
`1:N`; positions folded into `[0, L_i)`; `R_c ≤ min(L)/2` (enforced by the
constructor) so the minimum image is unambiguous.

See `spec/pairwise-correction.md`.
"""
function _correct_velocities_kernel!(
    V::AbstractMatrix{T},
    Y_sorted::AbstractMatrix{T},
    F_sorted::AbstractMatrix{T},
    cell_start::Vector{Int32},
    cell_end::Vector{Int32},
    original_index::Vector{Int32},
    num_cells::NTuple{3, Int32},
    L::NTuple{3, T},
    σ::T,
    Σ::T,
    a::T,
    μ::T,
    R_c::T,
) where {T}
    c_self = _self_correction(σ, Σ, a, μ)
    R_c² = R_c * R_c
    m_x, m_y, m_z = num_cells

    @inbounds for cz in Int32(0):(m_z - Int32(1))
        for cy in Int32(0):(m_y - Int32(1))
            for cx in Int32(0):(m_x - Int32(1))
                c = cx + (cy + cz * m_y) * m_x
                for s in cell_start[c + Int32(1)]:cell_end[c + Int32(1)]
                    Yn = SVector{3, T}(Y_sorted[1, s], Y_sorted[2, s], Y_sorted[3, s])
                    v = c_self *
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
                                    r² = x[1] * x[1] + x[2] * x[2] + x[3] * x[3]
                                    r² < R_c² || continue
                                    A, B = _correction_scalars(sqrt(r²), σ, Σ, μ)
                                    Fm = SVector{3, T}(
                                        F_sorted[1, s2], F_sorted[2, s2], F_sorted[3, s2],
                                    )
                                    xdotF = x[1] * Fm[1] + x[2] * Fm[2] + x[3] * Fm[3]
                                    v += A * Fm + (B * xdotF) * x
                                end
                            end
                        end
                    end
                    n = original_index[s]
                    V[1, n] += v[1]
                    V[2, n] += v[2]
                    V[3, n] += v[3]
                end
            end
        end
    end
    return V
end
