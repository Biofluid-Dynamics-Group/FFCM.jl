# Interpolation

Step 5 of the Fast FCM algorithm (Su & Keaveny 2024, §4).

## Summary

The fluid velocity $\boldsymbol{u}$ is set to match a no-slip condition on the particles. This
is realised by setting the particle velocities to be interpolations of the fluid velocity.
This operator,
$$\begin{align*}
    \tilde{\mathcal{J}} : \boldsymbol{L}^{2}\left(\Omega\right) &\to \left[\R^3\right]^N \\
    \boldsymbol{u} &\mapsto \int_{\Omega} \boldsymbol{u}\left(\boldsymbol{x}\right) \tilde{\Delta}_n\left(\boldsymbol{x}; \Sigma\right) \, \mathrm{d}\boldsymbol{x} \quad \text{(§3, equation(26))} \text{,}
\end{align*}$$
is the adjoint of the spreading operator $\mathcal{J}^\dagger$ because the spreading and interpolation
kernels are the same modified Gaussian kernel.

The resulting vector $\tilde{\mathcal{J}}\left[\boldsymbol{u}\right] = \left(\boldsymbol{V}_n\right)_{n = 1}^N$ corresponds to the particle velocities and closes the definition of the mobility operator.

Since we have a sampling of $\boldsymbol{u}$ on the gridpoints, the integral is approximated
by the trapezoidal rule on those points using the $M_G \times M_G \times M_G$ stencil. (REFACTOR NOTE: There should be a formal proof that this is required for the positive-definiteness, and a descriptive comment on why
it is spectrally accurate. Saying "Euler-Maclaurin" doesn't mean anything if you don't know that that is.)

In the code, we define
- `V` $= \left(\boldsymbol{V}_1, \dots, \boldsymbol{V}_n\right)$

## Why the quadrature is the trapezoidal rule, not Gauss quadrature

The paper specifies the trapezoidal rule (`outline.tex:185`). Three reasons
no higher-order quadrature (e.g. Gauss–Hermite) applies:

1. **The flow is only on the grid.** $\bm u$ is known on the uniform FFT
   grid alone. Gauss nodes fall between grid points and would require
   interpolating $\bm u$ first — extra error and cost for an
   already-discretised field.
2. **Trapezoidal is already spectral here.** On a periodic grid the
   trapezoidal rule is spectrally accurate for this smooth integrand
   (Euler–Maclaurin: periodicity cancels every boundary term).
3. **Adjointness / positive-definiteness.** The full mobility
   $\mathcal{M}^{\mathcal{V}\mathcal{F}} = \mathcal{J}\,\mathcal{L}^{-1}\,
   \mathcal{J}^\dagger$ is symmetric positive-definite *only because*
   interpolation is the exact discrete adjoint (transpose) of spreading
   (paper `outline.tex:324`). The trapezoidal rule with the uniform weight
   $\Delta x^3$ makes $\widetilde{\mathcal{J}} = \Delta x^3\,
   \widetilde{\mathcal{J}}^{\dagger\top}$ exactly; any other quadrature
   breaks the transpose and the SPD guarantee the downstream iterative
   resistance solver depends on.

Interpolation therefore reuses the spread's exact per-particle stencil
weights — a gather instead of a scatter, times $\Delta x^3$.

## Adjointness

Let $S$ be the spread matrix ($S_{g,n} = \widetilde{\Delta}_n(\bm x_g; \Sigma)$,
the action of `spread_forces!`) and $P$ the step-2 gather permutation
($F_\text{sorted} = P F$, `_gather_particles_kernel!`). Then

$$
\bm V = P^\top\, \Delta x^3\, S^\top\, \mathcal{L}^{-1}\, S\, P\, \bm F ,
\qquad
\mathcal{M}^{\mathcal{V}\mathcal{F}}
= \Delta x^3\, (S P)^\top\, \mathcal{L}^{-1}\, (S P),
$$

symmetric positive-definite because $\mathcal{L}^{-1}$ is. The scatter-back
`V[:, original_index[s]] = V_n` realises $P^\top$ — the transpose of the
step-2 gather. The interpolation weight at each stencil grid point is
identical to the spread weight; only the quadrature factor $\Delta x^3$ and
the gather-vs-scatter direction differ.

## Notation (paper-consistent)

In addition to the symbols pinned by [force-spreading.md](force-spreading.md)
and [stokes-solve.md](stokes-solve.md):

- `i ∈ {x, y, z}` (equivalently `{1, 2, 3}`) — Cartesian axis index.
- `n ∈ {1, …, N}` — particle index. `Y[i, n]`, `V[i, n]` map to paper
  $\bm Y_n$, $\widetilde{\bm V}_n$ (the `n`-th column).
- `Δ̃_n(x; Σ)` — the modified kernel, identical to the spread kernel
  (paper eq 267).
- `Δx³` — the trapezoidal-rule volume element on the uniform grid.

## Contract

### Cold-path input

No new cold-path inputs. Interpolation reuses the grid, kernel, and scratch
state built for steps 3 and 4.

### Cold-path derived state

No new `FFCMConfig` fields. Interpolation runs after spread + Stokes solve,
so the per-axis scratch vectors `gauss_x/y/z`, `r²_x/y/z`, `ind_x/y/z` (built
for step 3) are free to reuse.

### Hot-path input (per `interpolate_velocities!` call)

- `config.velocity_grid` — populated by `stokes_solve!` (step 4); the fluid
  velocity field $\bm u(\bm x_g)$ at every grid point.
- `config.Y_sorted` — sorted particle positions (step 2). Folded into
  $[0, L_i)$ by `wrap_positions!`.
- `config.original_index` — sorted slot `s` → original particle index `n`
  (step 2), used to scatter results back to caller order.

### Hot-path output

- `V::AbstractMatrix{T}` of shape `(3, N)`, **caller-owned, original particle
  order**. On return, `V[:, n]` holds $\widetilde{\bm V}_n$ for particle `n`
  in the caller's indexing. The terminal step produces the user-facing
  result, so it writes the output buffer directly rather than a `config`
  field — and `(3, N)` is the shape the matrix-free `mul!` pipeline needs.

### Side effects

- The scratch vectors `gauss_*`, `r²_*`, `ind_*` are overwritten with the
  last particle's per-axis precomputed values; they carry no between-call
  invariant (shared with step 3).
- `config.velocity_grid` is **not** modified (read-only consumer).

### Periodicity contract

Identical to spread: positions enter folded into $[0, L_i)$; the per-particle
stencil is wrapped mod $M_i$ inside the kernel; a particle near the domain
edge gathers from a stencil that wraps to the opposite side.

### Boundary cases

- **Particle exactly on a grid point** — distance 0 along each axis; no
  special case (same as spread).
- **`Σ = σ` degenerate limit** — $a_0 = 1$, $a_2 = 0$, so
  $\widetilde{\Delta}_n = \Delta_n$ and interpolation is the plain-Gaussian
  FCM volume average.

## API

```julia
"""
    interpolate_velocities!(V, config) -> V

Step 5 of the Fast FCM algorithm (Su & Keaveny 2024, §4 Step 5). Interpolate
the fluid velocity field `config.velocity_grid` (output of `stokes_solve!`)
to each particle position, writing the particle velocities into `V` (a `3×N`
matrix in the caller's original particle order). Reads `config.Y_sorted` and
`config.original_index` (populated by `sort_particles_by_cell!`).

Per particle the modified-kernel volume average
`Ṽ_n = ∫ u(x) Δ̃_n(x; Σ) d³x` (paper eq 283) is evaluated by the trapezoidal
rule over the same `M_G³` stencil as the spread, with weight `Δx³`. The
result is the exact discrete adjoint of `spread_forces!`, which makes the
assembled mobility operator symmetric positive-definite.

Allocation-free and type-stable on `T <: AbstractFloat`.

See `spec/interpolation.md`.
"""
function interpolate_velocities!(V::AbstractMatrix{T}, config::FFCMConfig{T}) where {T} end
```

The underscore-prefixed
`_interpolate_velocities_kernel!(V, velocity_grid, Y_sorted, original_index,
σ, Σ, Δx, inv_Δx, num_grid_points, M_G, gauss_x, gauss_y, gauss_z, r²_x, r²_y,
r²_z, ind_x, ind_y, ind_z)` is the function-barrier kernel: it takes naked
arrays and scalars so it is independently testable and benefits from Julia's
standard type-stability pattern. Mirrors the
`spread_forces!` / `_spread_forces_kernel!` split.

## Modified kernel — closed form

Identical to spread (paper eq 267, derived in
[force-spreading.md](force-spreading.md)): with $r = |\bm x - \bm Y_n|$ and
$\mathrm{pdmag} = \sigma^2 - \Sigma^2 \le 0$,
$$
\widetilde{\Delta}_n(\bm x; \Sigma) = (a_0 + a_2 r^2)\, \Delta_n(\bm x; \Sigma),
\qquad a_0 = 1 - \frac{3\,\mathrm{pdmag}}{2\Sigma^2},
\qquad a_2 = \frac{\mathrm{pdmag}}{2\Sigma^4}.
$$
The separable 1-D Gaussian weights, axis-squared distances, and
periodic-wrapped 1-based stencil indices are computed exactly as in spread
(nearest-anchored stencil $j_i = \mathrm{round}(Y_{n,i}\cdot\mathrm{inv}\Delta x)$,
`RoundNearestTiesToEven`).

The per-grid-point gather is
$$
\widetilde{\bm V}_n = \Delta x^3 \sum_{\text{stencil}} \bm u(\bm x_g)\,
(a_0 + a_2 r^2)\, g_x g_y g_z ,
$$
with the $\Delta x^3$ applied once per particle to the accumulated sum.

## Cold-path vs hot-path

| Phase | Allocations | Functions |
|---|---|---|
| Cold | — | No new cold-path work; reuses steps 3 and 4 state. |
| Hot  | `@ballocated == 0` | `interpolate_velocities!(V, config)` (after `spread_forces!` → `stokes_solve!`). |

The hot path is allocation-free and type-stable on `T <: AbstractFloat`.

## Performance notes

- **SoA stride-1 reads.** Destructure `(ux, uy, uz) = components(velocity_grid)`
  so the inner stencil loop reads `ux[ix, iy, iz]` stride-1 along $i_x$,
  mirroring the spread write loop.
- **Scalar accumulation.** Accumulate `vx, vy, vz` as scalars over the
  stencil; write the `SVector{3, T}` to `V[:, n]` once per particle, scaled
  by $\Delta x^3$.
- **`Δx³` applied once per particle**, not folded per grid point — 3 multiplies
  per particle versus 3 per stencil grid point. Algebraically identical;
  keeps the discrete-adjoint test bit-exact against the spread weights.
- **Scratch reuse.** The `gauss_*`, `r²_*`, `ind_*` vectors built for spread
  are reused; interpolation runs strictly after spread + solve, so there is
  no overlap.
- **No write race.** Unlike the spread scatter, the interpolation gather only
  reads the grid; the per-particle output columns are distinct (a
  permutation), so the loop is a clean future threading / SIMD target —
  deferred to a benchmark per `/julia-numerical-computing` (logged under
  PLAN.md CPU optimisation).

## Diffs from cuFCM

Audited `cuFCM/src/CUFCM_FCM.cu`
(`cufcm_particle_velocities_bpp_shared_dynamic`, lines 249–423) and
`cuFCM/src/CUFCM_SOLVER.cu` (`FCM_solver::gather`, lines 644–663). Only the
active (force-only, non-`USE_REGULARFCM`, `rotation == 0`) path is considered;
the commented `_tpp_register`, `_recompute`, `_selection`, `_mono` variants
are dead code per the cuFCM convention.

**cuFCM respects the adjoint.** Its spread kernel weight is
`g_x·g_y·g_z·(1 + temp3·r² − temp4)` with **no** `Δx³`
(`CUFCM_FCM.cu:39-47`); its interpolate weight is the *identical* term times
`norm = dx*dx*dx` (`CUFCM_FCM.cu:258,379-384`). So cuFCM implements
$\widetilde{\mathcal{J}} = \Delta x^3\,\widetilde{\mathcal{J}}^{\dagger\top}$
exactly — independent confirmation of the SPD constraint.

| Facet | cuFCM | This package | Decision |
|---|---|---|---|
| Quadrature weight | `norm = dx³` folded into the per-grid-point `temp1` | `Δx³` applied once per particle to the accumulated `(vx, vy, vz)` | **Keep** (diverge on location). Algebraically identical; cheaper and keeps the adjoint test bit-exact. |
| Stencil precompute | `xg = my_rint(Y/dx) − ngdh + (i mod ngd)`, `Anorm`, gauss, wrap — byte-identical to its spread kernel | identical; shared with spread via `_fill_particle_stencil!` | **Adopt** — reinforces the shared-helper extraction. |
| Polynomial factor | `1 + temp3·r² − temp4`, `temp3 = ½·pdmag/Σ⁴`, `temp4 = 3·½·pdmag/Σ²` | `(a₀ + a₂r²)`, `a₀ = 1 − 3·pdmag/2Σ²`, `a₂ = pdmag/2Σ⁴` | **Keep** — algebraically identical (same as spread). |
| r² handling | stores signed `xdis/ydis/zdis`, recomputes `r²` in the inner loop | stores `r²_*` directly | **Keep** the r²-only precompute (same divergence as spread). |
| Velocity-grid read | three SoA arrays `ux, uy, uz` (= `hx, hy, hz` reused), `ind = i + j·nx + k·nx·ny` | `components(velocity_grid)` → `ux, uy, uz`, column-major `(ix, iy, iz)` | **Adopt** — same SoA layout, `StructArray` wrapper. |
| Stencil reduction | `cub::BlockReduce.Sum` over threads (one block / particle, threads over `M_G³`) | serial scalar accumulation `vx += …` | **Keep** serial for the CPU MVP; block-reduce is the GPU pattern (revisit for CUDA). |
| Output / unsort | writes `VTEMP[3·np]` in **sorted** order; unsort handled by cuFCM's sort layer | writes `V[:, original_index[s]]` directly (inverse permutation folded in) | **Diverge.** Consistent with our step-2 decision to materialise sorted arrays; avoids an extra `V_sorted` buffer + pass. The fold is exactly $P^\top$, preserving adjointness. |
| start/end particle filter | `particle_index[np] ∈ [start, end)` sub-domain scheduling | none; full sorted range | **Out of scope** (same as spread). |
| Rotation/dipole (`W`) | `rotation == 1`: angular velocity $\tfrac12\nabla\times\bm u$ weighted by grad-Gaussian → `WTEMP` | none — force-only $\mathcal{M}^{\mathcal{V}\mathcal{F}}$ | **Out of scope** (PLAN.md). |
| `USE_REGULARFCM` | compile-time `Σ = σ` branch with separate `sigmadip` | no flag; `Σ_over_σ = 1` degenerate limit subsumes it | **Subsume via degenerate limit** (same as spread). |
| Rounding / wrap | `my_rint` (ties-even); `xg − n·floor(xg/n)` | `round(…, RoundNearestTiesToEven)`; `mod(g, M) + 1` | **Keep** — semantically identical (same as spread). |

**GPU-revisit note** (future CUDA backend): interpolate is the gather mirror
of spread's scatter — one block per particle, threads over the `M_G³` stencil,
`cub::BlockReduce` to sum. Unlike spread's `atomicAdd`, the gather has no write
race. Logged alongside the spread/CUDA concurrency note in `PLAN.md`.

## Verification

Test files follow the flat-`test/` layout and the domain-language naming
convention.

**Tolerance convention.** Relative comparisons to a non-zero reference use
`rtol = sqrt(eps(T))`; quantities that should be `≈ 0` or arrays mixing large
and near-zero entries use `atol = 1e-10` (`Float64`) / `1e-6` (`Float32`)
(≈ `rtol/100`); truncation-dominated tests (constant-flow, single-sphere drag)
use the documented `(M_G, Σ/Δx)` / finite-box tolerance stated inline.

`test/test_interpolate_velocities.jl`:

1. **Output shape and scatter-back order.** `V` is `(3, N)`; with particles
   supplied in scrambled order, `V` comes out in original (caller) order —
   verified against the `original_index` permutation.
2. **Discrete adjoint of spreading.** `⟨interpolate(u), eₙ⟩ = Δx³·⟨u, spread(eₙ)⟩`
   for a chosen grid field `u` and each particle `n`. Pins
   $\widetilde{\mathcal{J}} = \Delta x^3\, S^\top$.
3. **Mobility symmetry.** `Fₐᵀ M Fᵦ = Fᵦᵀ M Fₐ` for random `Fₐ, Fᵦ` through the
   full spread → solve → interpolate pipeline. Pins the SPD precondition
   (`outline.tex:324`).
4. **Single-particle closed form.** One particle at a known position; set
   `velocity_grid` to a known analytic field; hand-compute
   `V_n = Σ_stencil u(x_g)·Δ̃_n(x_g)·Δx³`.
5. **Constant flow → constant velocity.** Uniform `velocity_grid = U` ⇒
   `V_n ≈ U` for every particle (the kernel integrates to 1), within the
   `(M_G, Σ/Δx)` truncation tolerance (paper Table 2).
6. **Linearity in the flow field.**
   `interpolate(α u₁ + β u₂) = α·interpolate(u₁) + β·interpolate(u₂)`.
7. **Periodicity.** Particle at `Yₙ` vs `Yₙ + L êᵢ` (after `wrap_positions!`)
   on the same `velocity_grid` ⇒ identical `V_n`.
8. **Standard-FCM degenerate limit (Σ = σ).** `Σ_over_σ = 1` ⇒ interpolation
   uses the plain Gaussian (`a₀ = 1, a₂ = 0`), matching the closed-form FCM
   volume average.

`test/test_single_sphere_mobility.jl`:

9. **Single-sphere periodic self-mobility (end-to-end).** One particle, force
   `F`, `Σ/σ = 1`, full pipeline `wrap → assign → sort → spread → solve →
   interpolate`. Compare `V` to the continuum periodic self-mobility
   $\widetilde{\bm V}_\text{self} = \tfrac{1}{L^3}\sum_{\bm k \ne 0}
   \tfrac{e^{-\sigma^2 k^2}}{\mu k^2}(\bm I - \hat{\bm k}\hat{\bm k}^\top)\bm F$
   (the Fourier-space form of the regularised Stokeslet, paper eq 205–207,
   convolved with the kernel on both sides), summed over the reciprocal
   lattice $\bm k = (2\pi/L)\bm n$ up to where $e^{-\sigma^2 k^2}$ falls below
   a documented cutoff. Tolerance is the grid-discretisation / aliasing error
   for the chosen `Σ/Δx`, documented inline. First test exercising all five
   steps composed.

`test/test_interpolate_velocities_inferred.jl` — `@inferred` for
`interpolate_velocities!` and `_interpolate_velocities_kernel!`,
`Float32`/`Float64`.

`test/test_interpolate_velocities_allocations.jl` — `@ballocated == 0` for the
wrapper and the kernel.

`test/test_jet.jl` — `JET.@test_call interpolate_velocities!` walks the call
graph for inference health.

`test/test_aqua.jl` — `Aqua.test_all` covers module hygiene (unchanged).
