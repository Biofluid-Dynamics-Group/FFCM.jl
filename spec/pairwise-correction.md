# Pairwise correction

Step 6 of the Fast FCM algorithm (Su & Keaveny 2024, §4 Step 6;
`paper/tex/outline.tex:533`), the real-space part of the Ewald-style splitting
introduced in paper §4 (`outline.tex:246-250`),

$$
\mathcal{M}^{\mathcal{V}\mathcal{F}}
= \widetilde{\mathcal{M}}^{\mathcal{V}\mathcal{F}}
+ \left(\mathcal{M}^{\mathcal{V}\mathcal{F}} - \widetilde{\mathcal{M}}^{\mathcal{V}\mathcal{F}}\right).
$$

Steps 3–5 evaluate the grid part $\widetilde{\mathcal{M}}^{\mathcal{V}\mathcal{F}}$
with the modified kernel of width $\Sigma > \sigma$ (paper eq 267). Because
$\Sigma \ne \sigma$, the grid result carries a near-field error. Step 6 adds the
analytic correction $\mathcal{M}^{\mathcal{V}\mathcal{F}} -
\widetilde{\mathcal{M}}^{\mathcal{V}\mathcal{F}}$, which decays exponentially in
particle separation (`outline.tex:318`) and is therefore applied as a **sparse,
real-space pairwise sum** over the step-1/2 cell list, plus a per-particle **self
term**. After step 6, the assembled operator returns the true $\sigma$-regularised
mobility — **independent of the chosen $\Sigma$**, which is the central correctness
property of the splitting.

Scope is **force → velocity ($\mathcal{M}^{\mathcal{V}\mathcal{F}}$) only**, matching
steps 1–5; the torque/angular-velocity corrections (paper eq:correction_WF,
eq:correction_WT) are out of scope.

## The correction is the difference of two FCM pairwise mobilities

The FCM pairwise mobility relating the force on particle $m$ to the velocity of
particle $n$ is the regularised Stokeslet at width $\sigma\sqrt2$ (paper eq:M_VF,
`outline.tex:221`):

$$
\bm{M}^{\mathcal{V}\mathcal{F}}_{nm} = \bm{S}(\bm{Y}_n - \bm{Y}_m; \sigma\sqrt2),
$$

and the modified-kernel mobility expands (paper `outline.tex:302`) as

$$
\widetilde{\bm{M}}^{\mathcal{V}\mathcal{F}}_{nm}
= \bm{S}(\Sigma\sqrt2) + (\sigma^2 - \Sigma^2)\bm{Q}(\Sigma\sqrt2)
+ \tfrac{(\sigma^2 - \Sigma^2)^2}{4}\bm{T}(\Sigma\sqrt2),
$$

with $\bm{S} = \bm{S}^{(1)} + \bm{S}^{(2)} + \bm{S}^{(3)}$ (paper eq:S1–S3,
`outline.tex:205-207`) and $\bm{Q} = \bm{Q}^{(1)} + \bm{Q}^{(2)}$ (paper eq:Q1–Q2,
`outline.tex:235-236`), and $\bm{T}$ given by `outline.tex:306`. Their difference is
the correction (paper eq:correction_VF, `outline.tex:310-316`).

### Implementation form (paper-simplified, typo-corrected)

The package evaluates the paper's regrouped closed form directly. With
$\bm{x} = \bm{Y}_n - \bm{Y}_m$, $r = \lVert\bm{x}\rVert$, $\eta = \mu$,
$\mathrm{erf}_{2w} \equiv \mathrm{erf}(r/2w)$, and the Gaussian at the $\sqrt2$-scaled
width $\Delta_w \equiv \Delta(\bm{x}; w\sqrt2) = (4\pi w^2)^{-3/2}\exp(-r^2/4w^2)$:

$$
\bm{M}^{\mathcal{V}\mathcal{F}}_{nm} - \widetilde{\bm{M}}^{\mathcal{V}\mathcal{F}}_{nm}
= (\bm{G} + \sigma^2\nabla^2\bm{G})(\mathrm{erf}_{2\sigma} - \mathrm{erf}_{2\Sigma})
+ \bm{S}^{(3)}(\sigma\sqrt2) - \bm{S}^{(3)}(\Sigma\sqrt2)
- (\sigma^2 - \Sigma^2)\bm{Q}^{(2)}(\Sigma\sqrt2)
- \tfrac{(\sigma^2 - \Sigma^2)^2}{4}\bm{T}(\Sigma\sqrt2).
$$

This is marginally leaner than evaluating the two full mobilities and subtracting:
the $(\bm{G} + \sigma^2\nabla^2\bm{G})$ grouping analytically pre-cancels
$\bm{Q}^{(1)}$'s `erf`, so only $\bm{Q}^{(2)}$ and $\bm{T}$ (both Gaussian-only)
survive as explicit terms. The transcendental cost is the **same** either way —
two `erf` and two `exp` per interaction.

> ### ⚠️ Paper typo flagged for future inspection
>
> Paper eq:correction_VF (`outline.tex:312`) prints the first factor as
> $\bigl(\mathrm{erf}(r/(\sigma\sqrt2)) - \mathrm{erf}(r/(\Sigma\sqrt2))\bigr)$. The
> regrouping closes **only** with $\mathrm{erf}(r/(2\sigma))$, $\mathrm{erf}(r/(2\Sigma))$
> — that is, $\mathrm{erf}(r/(s\sqrt2))$ evaluated at the $\sqrt2$-scaled width
> $s = \sigma\sqrt2$, which is the argument used everywhere else in the paper
> (eq:S1, eq:M_VF) and in the author's reference implementation
> (`cuFCM/src/CUFCM_CORRECTION.cu:154,157`). Derivation: collecting the $\nabla^2\bm{G}$
> terms from $\bm{S}^{(2)}(\sigma\sqrt2) - \bm{S}^{(2)}(\Sigma\sqrt2) -
> (\sigma^2-\Sigma^2)\bm{Q}^{(1)}(\Sigma\sqrt2)$ gives the coefficient
> $\sigma^2(\mathrm{erf}_{2\sigma} - \mathrm{erf}_{2\Sigma})$, fixing the argument as
> $r/(2w)$. We therefore implement $r/(2w)$ and **flag the printed line 312 as a
> suspected typo** to revisit against the published article. The implementation is
> independently cross-checked against the difference-of-full-mobilities form (the
> verification §, test 2), so a wrong argument would fail the suite.

### Tensor coefficients and the two pair scalars

Each building block has the form $c_I\,\bm{I} + c_{xx}\,\bm{x}\bm{x}^\top$ with
**un-normalised** $\bm{x}\bm{x}^\top$ (avoids a `/r²`; matches cuFCM). The verified
coefficients (paper eq:S3 `:207`, $\bm{G}$ `:211`, eq:Q2 `:236`, $\bm{T}$ `:306`;
$\bm{Q}^{(2)}, \bm{T}$ carry no `erf`):

| block | $c_I$ | $c_{xx}$ |
|---|---|---|
| $\bm{G}$ | $\dfrac{1}{8\pi\eta r}$ | $\dfrac{1}{8\pi\eta r^3}$ |
| $\sigma^2\nabla^2\bm{G}$ | $\dfrac{\sigma^2}{4\pi\eta r^3}$ | $-\dfrac{3\sigma^2}{4\pi\eta r^5}$ |
| $\bm{S}^{(3)}(w\sqrt2)$ | $-\dfrac{2w^4}{\eta r^2}\Delta_w$ | $+\dfrac{6w^4}{\eta r^4}\Delta_w$ |
| $\bm{Q}^{(2)}(\Sigma\sqrt2)$ | $-\dfrac{1}{\eta}\bigl(1+\tfrac{2\Sigma^2}{r^2}\bigr)\Delta_\Sigma$ | $+\dfrac{1}{\eta r^2}\bigl(1+\tfrac{6\Sigma^2}{r^2}\bigr)\Delta_\Sigma$ |
| $\bm{T}(\Sigma\sqrt2)$ | $\dfrac{1}{2\eta\Sigma^2}\bigl(2-\tfrac{r^2}{2\Sigma^2}\bigr)\Delta_\Sigma$ | $\dfrac{1}{4\eta\Sigma^4}\Delta_\Sigma$ |

The correction collapses to **two scalars** $A(r)$ (coefficient of $\bm{I}$) and
$B(r)$ (coefficient of $\bm{x}\bm{x}^\top$); with $pd = \sigma^2 - \Sigma^2$:

$$
A(r) = (\mathrm{erf}_{2\sigma}-\mathrm{erf}_{2\Sigma})\Bigl(\tfrac{1}{8\pi\eta r}+\tfrac{\sigma^2}{4\pi\eta r^3}\Bigr)
- \tfrac{2\sigma^4}{\eta r^2}\Delta_\sigma
+ \Bigl[\tfrac{2\Sigma^4}{\eta r^2}+\tfrac{pd}{\eta}\bigl(1+\tfrac{2\Sigma^2}{r^2}\bigr)-\tfrac{pd^2}{4}\tfrac{2-r^2/2\Sigma^2}{2\eta\Sigma^2}\Bigr]\Delta_\Sigma ,
$$
$$
B(r) = (\mathrm{erf}_{2\sigma}-\mathrm{erf}_{2\Sigma})\Bigl(\tfrac{1}{8\pi\eta r^3}-\tfrac{3\sigma^2}{4\pi\eta r^5}\Bigr)
+ \tfrac{6\sigma^4}{\eta r^4}\Delta_\sigma
+ \Bigl[-\tfrac{6\Sigma^4}{\eta r^4}-\tfrac{pd}{\eta r^2}\bigl(1+\tfrac{6\Sigma^2}{r^2}\bigr)-\tfrac{pd^2}{4}\tfrac{1}{4\eta\Sigma^4}\Bigr]\Delta_\Sigma .
$$

The velocity correction to particle $n$ from the force $\bm{F}_m$ on particle $m$ is
then

$$
\Delta\bm{V}_n \mathrel{+}= A(r)\,\bm{F}_m + B(r)\,(\bm{x}\cdot\bm{F}_m)\,\bm{x}.
$$

Because $A, B$ depend only on $r$ and the tensor is symmetric, the pair correction
is self-adjoint ($\bm{M}_{nm} = \bm{M}_{mn}$), so the splitting preserves the
symmetric-positive-definite structure of the mobility (paper §4.3,
`outline.tex:320-324`).

### Self term

At $r = 0$ the correction is the well-defined diagonal (paper eq:correction_VF_limit,
`appendix.tex:52-54`), a scalar times $\bm{I}$ added to every particle independent of
its neighbours, with $a = \sigma\sqrt\pi$ (paper eq 163):

$$
c_\mathrm{self} = \frac{1}{6\pi\eta a} - \frac{1}{6\pi\eta(\Sigma\sqrt\pi)}
+ \frac{\sigma^2-\Sigma^2}{12\eta(\Sigma\sqrt\pi)^3}
- \frac{(\sigma^2-\Sigma^2)^2}{32\eta\,\Sigma^5\pi^{3/2}}.
$$

At $\Sigma = \sigma$ both $c_\mathrm{self}$ and every pair term vanish, so the
correction is identically zero — the standard-FCM degenerate limit, consistent with
$\widetilde{\mathcal{M}}^{\mathcal{V}\mathcal{F}} = \mathcal{M}^{\mathcal{V}\mathcal{F}}$.

## Notation (paper-consistent)

In addition to the symbols pinned by earlier specs:

- `i ∈ {x, y, z}` — Cartesian axis index. `n, m ∈ {1, …, N}` — particle indices.
- `x = Y_n − Y_m` — minimum-image separation; `r = ‖x‖`.
- `σ`, `Σ` — FCM and modified-kernel widths (`config.σ`, `config.Σ`). `a = σ√π`.
- `μ` — viscosity (paper `η`). `R_c` — pairwise cutoff (`config.R_c`).
- `A(r)`, `B(r)` — pair scalars (coefficients of `I` and `xxᵀ`).
- `c_self` — the `r → 0` self-correction scalar.

## Contract

### Cold-path input

No new cold-path inputs. The correction reuses the cell geometry (`R_c`,
`num_cells`, `cell_size`) and kernel widths (`σ`, `Σ`, `a`, `μ`) already on
`FFCMConfig`.

### Cold-path derived state

No new `FFCMConfig` fields. `c_self` is recomputed once per `correct_velocities!`
call (≈ 8 flops, negligible beside the pair loop) by `_self_correction`, keeping the
struct unchanged and the closed form in one directly-testable function.

The constructor gains one validation: `R_c ≤ min(L)/2`, so the minimum-image
separation is unambiguous and no particle is corrected against its own periodic
image (paper assumption, `outline.tex:318`).

### Hot-path input (per `correct_velocities!` call)

- `config.Y_sorted`, `config.F_sorted` — sorted positions and forces (step 2).
- `config.cell_start`, `config.cell_end` — per-cell sorted-slot ranges (step 2).
- `config.original_index` — sorted slot `s` → original particle index `n`.
- Positions folded into `[0, L_i)` by `wrap_positions!` (step 1).

### Hot-path output

- `V::AbstractMatrix{T}` of shape `(3, N)`, **caller-owned, original particle
  order** — the same buffer `interpolate_velocities!` wrote. `correct_velocities!`
  **adds** the correction in place (`V[:, n] += ΔV_n`), realising
  $\mathcal{M} = \widetilde{\mathcal{M}} + (\mathcal{M} - \widetilde{\mathcal{M}})$.

### Side effects

- None beyond writing `V`. `Y_sorted`, `F_sorted`, and the cell list are read-only;
  no scratch buffers are touched (the per-particle accumulation lives in registers /
  `SVector`s).

### Periodicity contract

Each pair interacts through its nearest periodic image only: the separation
`Y_n − Y_m` is reduced to `[−L_i/2, L_i/2]` per axis by `_min_image`. The 27-cell
neighbour sweep wraps cell coordinates mod `num_cells[i]`. The `R_c ≤ min(L)/2`
precondition guarantees a single relevant image and excludes self-image
corrections.

### Boundary cases

- **No neighbours within `R_c`** — only the self term is added (`V[:, n] += c_self·F_n`).
- **`Σ = σ` degenerate limit** — `A = B = 0` and `c_self = 0`; `V` is unchanged.
- **Coincident distinct particles (`r → 0`)** — assumed not to occur (the paper
  forbids self-image corrections; physical configurations keep particles apart). The
  pair scalars are evaluated only for `0 < r < R_c`; the `r = 0` case is the separate
  self term.

## API

```julia
"""
    correct_velocities!(V, config) -> V

Step 6 of the Fast FCM algorithm (Su & Keaveny 2024, §4 Step 6). Add the
real-space pairwise correction (M^VF − M̃^VF) and the per-particle self term to
the interpolated velocities `V` (a `3×N` matrix in the caller's original particle
order — the same buffer `interpolate_velocities!` wrote). Reads `config.Y_sorted`,
`config.F_sorted`, the cell list (`config.cell_start` / `config.cell_end`), and
`config.original_index` (populated by `sort_particles_by_cell!`).

Each particle gathers the correction from neighbours within `config.R_c` over the
27 surrounding cells (minimum-image), applying `ΔV_n += A(r)·F_m + B(r)·(x·F_m)·x`
with the two pair scalars of paper eq:correction_VF, plus the closed-form self term
of eq:correction_VF_limit. The result completes the σ-regularised mobility,
independent of Σ.

Allocation-free and type-stable on `T <: AbstractFloat`.

See `spec/pairwise-correction.md`.
"""
function correct_velocities!(V::AbstractMatrix{T}, config::FFCMConfig{T}) where {T} end
```

The function-barrier kernel
`_correct_velocities_kernel!(V, Y_sorted, F_sorted, cell_start, cell_end,
original_index, num_cells, L, σ, Σ, a, μ, R_c)` takes naked arrays and scalars so it
is independently testable and fully type-stable; it mirrors the
`interpolate_velocities!` / `_interpolate_velocities_kernel!` split. Two pure
helpers carry the physics: `_correction_scalars(r, σ, Σ, μ) -> (A, B)` and
`_self_correction(σ, Σ, a, μ) -> c_self`. `_min_image(x, L)` reduces a separation
`SVector{3}` to the nearest image.

## Cold-path vs hot-path

| Phase | Allocations | Functions |
|---|---|---|
| Cold | — | `R_c ≤ min(L)/2` validation in the constructor; no new fields. |
| Hot  | `@ballocated == 0` | `correct_velocities!(V, config)` (after `interpolate_velocities!`). |

The hot path is allocation-free and type-stable on `T <: AbstractFloat`.

## Performance notes

- **Gather, not half-list — flagged tradeoff.** Each particle accumulates its own
  correction by reading every neighbour within `R_c` over the full 27 cells, writing
  only its own output column. This evaluates each ordered pair's scalars twice
  (≈ 2× the `erf`/`exp` of a half-neighbour list), but it has **no write race**:
  threads/SIMD parallelise trivially per particle, exactly like
  `interpolate_velocities!`, and it matches the paper's CUDA block-per-particle
  *gather* strategy (`appendix.tex:67,71`). cuFCM instead uses a 13-neighbour half
  list with Newton's-third-law dual update and `atomicAdd` on the partner
  (`CUFCM_CORRECTION.cu:226-321`) — fewer evaluations, but a write race. We revisit
  the half-list (or thread-local accumulators) only if benchmarks show the pair loop
  dominates (logged in PLAN.md).
- **Two-scalar collapse.** `A·I + B·xxᵀ` (2 scalars) instead of a dense 3×3 matvec;
  the $\Sigma\sqrt2$ `erf`/Gaussian are shared across the $\bm{Q}^{(2)}, \bm{T}$
  terms, so only two `erf` and two `exp` are evaluated per interaction.
- **Self term once per call.** `c_self` is computed once at kernel entry, not stored
  on `config` and not recomputed per particle.
- **Contiguous cell walk.** The outer loop iterates cells in increasing linear hash
  (`z` slowest, `x` fastest), so `Y_sorted`/`F_sorted` columns are read in their
  sorted (contiguous-per-cell) order; the per-particle output is scattered to
  `V[:, original_index[s]]`.
- **`@inbounds` after a once-per-call bounds argument**; the inner cutoff branch
  (`r² < R_c²`) blocks naive `@simd`, so vectorisation is deferred to a benchmark
  (per `/julia-numerical-computing`).

## Diffs from cuFCM

Audited `cuFCM/src/CUFCM_CORRECTION.cu` (`cufcm_pair_correction`,
`cufcm_self_correction`, the `S_I/S_xx/Q_I/Q_xx/T_I/T_xx` coefficient helpers) and
`cuFCM/src/CUFCM_SOLVER.cu:88-120` (the self-term constants and cell sizing). Only the
active force-only (`ROTATION == 0`) path is considered; the `#if ROTATION` blocks and
the commented `cufcm_compute_formula` variant are out of scope.

| Facet | cuFCM | This package | Decision |
|---|---|---|---|
| Correction formula | Difference of full mobilities `S(σ√2) − S(Σ√2) − (σ²−Σ²)Q(Σ√2) − ¼(σ²−Σ²)²T(Σ√2)` (`CORRECTION.cu:171-181`) | Paper-simplified eq:correction_VF (typo-corrected); cuFCM form is the **test oracle** | **Keep** the simplified form (`Q⁽¹⁾`'s `erf` pre-cancelled, marginally leaner). Same 2 `erf` + 2 `exp` cost; cross-checked against cuFCM's form (test 2). |
| `erf` argument | `erf(r/(2σ))`, `erf(r/(2Σ))` (= `erf(r/(s√2))`, `s = σ√2`) | Same | **Adopt** — the correct argument; paper line 312 typo flagged above. |
| Tensor decomposition | `c_I·I + c_xx·xxᵀ`, **un-normalised** `xxᵀ` | Same | **Adopt** — one fewer division than `x̂x̂ᵀ`. |
| Viscosity | `η = 1` (omitted from the helpers) | `μ` carried explicitly | **Keep** — matches `stokes_solve!`/`config`; `1/μ` reinstated in the coefficients. |
| Self term | `V += F·(Stokes − Mod + PD − BiLap)` (`CORRECTION.cu:343`; constants `SOLVER.cu:88-98`) | `_self_correction` closed form | **Adopt** — confirmed equal to eq:correction_VF_limit term-for-term. |
| Neighbour search | 13-neighbour **half** cell list (`map[13·icell]`) + Newton's-third-law dual update with `atomicAdd` (`CORRECTION.cu:226-321`) | Full **27**-cell **gather**, self-write only, no atomics | **Diverge** — see performance notes (no write race; GPU-gather-aligned). |
| Min-image | `x − L·int(x/(0.5L))` (truncation), per pair | `x − L·round(x/L)`, per pair | **Keep** — equivalent for `r < L/2`. |
| Cutoff test / index | `rijsq < Rrefsq`; `int` indices; `Rc = Rc_fac·dx` | `r² < R_c²`; `Int32`; `R_c` a direct user parameter | **Keep** — our conventions; `R_c` calibration deferred (PLAN.md). |
| Cell count | `max(L/Rc, 3)` | `max(floor(L/R_c), 3)` (`config.jl:99`) | **Keep** — already matches. |
| Grid-based correction variant | `cufcm_flowfield_correction` (`CORRECTION.cu:356-493`) spreads a correction force onto the grid | — | **Out of scope** — not the paper's pairwise method; also carries an apparent `S_xx` argument bug (`:481`). |
| Rotation/torque | `#if ROTATION` → M^ΩF / M^ΩT via `f`, `K`, `P` helpers and `W`/`T` arrays | — | **Out of scope** — force → velocity only. |

**GPU-revisit note** (future CUDA backend): the paper assigns one thread per particle
to the correction and gathers (`appendix.tex:71`); our 27-cell gather maps directly,
unlike cuFCM's half-list + `atomicAdd`. Logged in PLAN.md alongside the spread/CUDA
concurrency note.

## Verification

Test files follow the flat-`test/` layout and the domain-language naming convention.

**Tolerance convention.** Relative comparisons to a non-zero reference use
`rtol = sqrt(eps(T))`; quantities that should be `≈ 0` or arrays mixing large and
near-zero entries use `atol = 1e-10` (`Float64`) / `1e-6` (`Float32`); the
truncation-dominated end-to-end test uses the documented `(M_G, Σ/Δx)` / finite-box
tolerance stated inline.

`test/test_correct_velocities.jl`:

1. **Self term matches the `r → 0` limit.** `_self_correction(σ, Σ, a, μ)` equals the
   hand-evaluated eq:correction_VF_limit (`appendix.tex:52-54`) over a sweep of
   `Σ/σ` and `μ`.
2. **Pair scalars reproduce the difference of full mobilities.** For sampled `x`,
   assemble `S(σ√2) − S(Σ√2) − (σ²−Σ²)Q(Σ√2) − ¼(σ²−Σ²)²T(Σ√2)` as a `3×3 SMatrix`
   from the full paper tensors (eq:S1–S3, eq:Q1–Q2, eq:T) and assert it equals
   `A(r)·I + B(r)·xxᵀ`. An algebraically-distinct oracle that pins the `A, B` collapse
   and the line-312 `erf` argument.
3. **Standard-FCM degenerate limit (`Σ = σ`).** `_self_correction` and
   `_correction_scalars` both return 0; `correct_velocities!` leaves `V` unchanged.
4. **Isolated particle gets only the self term.** One particle (or particles spaced
   beyond `R_c`): `V[:, n] += c_self·F_n` exactly.
5. **Two-particle pair correction matches the tensor.** Two particles at `r < R_c`:
   the added velocity equals the difference-of-mobilities tensor (test 2) times the
   force, for both partners.
6. **Pair correction is symmetric (SPD precondition).** `Fₐᵀ (corr Fᵦ) = Fᵦᵀ (corr Fₐ)`
   through the operator, pinning $\bm{M}_{nm} = \bm{M}_{mn}^\top$ (`outline.tex:320-324`).
7. **Linearity in F**, and **translation + periodicity invariance** — shifting all
   positions, or one particle by `L êᵢ`, leaves the correction unchanged.
8. **Additive onto `V`.** `correct_velocities!` adds to a pre-filled `V` rather than
   overwriting it.

`test/test_single_sphere_mobility.jl` (extended):

9. **Σ-independent single-sphere self-mobility (end-to-end, all six steps).** Run
   `wrap → assign → sort → spread → solve → interpolate → correct` for one particle
   at `Σ/σ ∈ {1.5, 3}`. The corrected self-velocity matches the σ-regularised
   reciprocal-lattice Stokeslet sum
   $\tfrac{1}{L^3}\sum_{\bm k \ne 0} \tfrac{e^{-\sigma^2 k^2}}{\mu k^2}(\bm I - \hat{\bm k}\hat{\bm k}^\top)\bm{F}$
   (the same reference as the step-5 test), and the two `Σ` runs agree to the
   documented grid-truncation tolerance. This is the proof that the correction
   removes the Σ-dependence introduced by the grid step.

`test/test_correct_velocities_inferred.jl` — `@inferred` for `correct_velocities!`,
`_correct_velocities_kernel!`, `_correction_scalars`, `_self_correction`,
`Float32`/`Float64`.

`test/test_correct_velocities_allocations.jl` — `@ballocated == 0` for the wrapper
and the kernel.

`test/test_jet.jl` — `JET.@test_call correct_velocities!` walks the call graph for
inference health.

`test/test_aqua.jl` — `Aqua.test_all` covers module hygiene (now including the
`SpecialFunctions` dependency).
