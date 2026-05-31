# Stokes Solve

Step 4 of the Fast FCM algorithm (Su & Keaveny 2024, §4).

## Summary

The continuous Stokes problem is now
$$\begin{align*}
  & -\mu \Delta \boldsymbol{u} + \nabla p
  = \boldsymbol{f} &\quad& \text{in} \; \Omega \quad \text{(§3, equation (23))} \; \text{and}\\
  & \operatorname{div}\left(\boldsymbol{u}\right) = 0 &\quad& \text{in} \; \Omega \quad \text{(§3, equation (24))}\text{,}
\end{align*}$$
with periodic boundary conditions, where $\boldsymbol{f} = \tilde{\mathcal{J}}^\dagger\left[\left\{ \boldsymbol{F}_n \right\}_{n = 1}^N\right]$. A spectral method for this problem corresponds to solving
the algebraic equation that arises from applying the Fourier transform to the momentum balance equation.

Solving for the Fourier transform of the velocity, one gets
$$
  \hat{\boldsymbol{u}}(\boldsymbol{k}) = \frac{1}{\mu \lvert \boldsymbol{k} \rvert^2} \left(\boldsymbol{I} - \frac{\boldsymbol{k} \otimes \boldsymbol{k}}{\lvert \boldsymbol{k} \rvert^2}\right) \hat{\boldsymbol{f}}(\boldsymbol{k}) \text{,}
$$
for $\boldsymbol{k} \neq 0$ and $\hat{\boldsymbol{u}}(\boldsymbol{0}) = \boldsymbol{0}$. Note that
$\boldsymbol{k}$ is the wavenumber vector as this is a 3D Fourier transform.

The spectral method is then to apply the FFT to the spread forces sampled on the grid to obtain $\hat{\boldsymbol{f}}$, solve for $\hat{\boldsymbol{u}}$, and apply the inverse FFT to recover $\boldsymbol{u}$ sampled on the grid.

In the code, we define
- `\mu` $= \mu$,
- `f[i, j, k]` $= \boldsymbol{f}(i h_x, j h_y, k h_z)$, (REFACTOR NOTE: This is now `force_grid`, the naming needs to change)
- `u[i, j, k]` $= \boldsymbol{u}(i h_x, j h_y, k h_z)$, (REFACTOR NOTE: This is now `velocity_grid`, the naming needs to change)
- `fluid_hat[q, r, s]` $= \hat{\boldsymbol{f}}(k_q, k_r, k_s)$ initially, rewritten to $\hat{\boldsymbol{u}}(k_q, k_r, k_s)$ after the solve.


## Notation

In addition to the symbols already pinned by
[force-spreading.md](force-spreading.md):

- $\mu$ — fluid dynamic viscosity (paper eq 152,
  `outline.tex:152`). **Convention note:** the paper writes $\eta$ for
  this same quantity; this package writes $\mu$. Code identifier
  `μ::T`.
- $\Delta$ — the Laplacian, $\Delta \bm u \equiv \partial_i \partial_i \bm u$
  (Einstein summation). **Convention note:** the paper writes
  $\nabla^2$ for this same operator; this package writes $\Delta$.
  The kernel symbol $\Delta_n(\bm x; \sigma)$ stays subscripted, and
  the grid spacing $\Delta x$ stays axis-qualified — neither collides
  with the unsubscripted Laplacian.
- $\bm k$ — wavenumber on the FFT grid; per axis
  $k_i = 2\pi n_i / L_i$ with $n_i$ the signed FFT index.
- $\hat{\bm f}(\bm k)$, $\hat{\bm u}(\bm k)$ — Fourier transforms of
  the spread force and the resulting velocity.
- $M_\text{total} = M_x \cdot M_y \cdot M_z$ — total grid point count;
  used for FFTW normalisation.

## Contract

### Cold-path input (new in this step)

- `μ::T` — required keyword on `FFCMConfig`. Must satisfy `μ > 0`,
  else `ArgumentError`. No default; viscosity is a physical quantity
  and should be an explicit caller decision.

The existing cold-path inputs from steps 1–3 (`L`, `R_c`, `N`, `a`,
`Σ_over_σ`, `num_grid_points`, `M_G`) are unchanged.

### Cold-path derived state (added to `FFCMConfig`)

| Field | Type | Definition |
|---|---|---|
| `μ` | `T` | The user input. |
| `velocity_grid` | `StructArray{SVector{3, T}, 3, …}` | Shape `(M_x, M_y, M_z)`. Same SoA layout as `force_grid`. Holds $\bm u(\bm x_g)$ after `stokes_solve!`. |
| `fluid_hat` | `StructArray{SVector{3, Complex{T}}, 3, …}` | Shape `(M_x ÷ 2 + 1, M_y, M_z)`. Three backing `Array{Complex{T}, 3}`. Holds $\hat{\bm f}(\bm k)$ on entry to the projection and $\hat{\bm u}(\bm k)$ on exit. |
| `k_x` | `Vector{T}` | Length `M_x ÷ 2 + 1`. Wavenumbers along the r2c half-axis: `k_x[i] = 2π · (i - 1) / L_x`. |
| `k_y` | `Vector{T}` | Length `M_y`. FFTW wrap-around: `k_y[j] = 2π/L_y · (j ≤ M_y÷2+1 ? j - 1 : j - 1 - M_y)`. |
| `k_z` | `Vector{T}` | Length `M_z`. Analogous to `k_y`. |
| `forward_plan` | concrete FFTW r2c plan | Built once at cold path; applies to any `Array{T, 3}` of shape `(M_x, M_y, M_z)`. |
| `backward_plan` | concrete FFTW c2r plan | Built once at cold path; applies to any `Array{Complex{T}, 3}` of shape `(M_x ÷ 2 + 1, M_y, M_z)`. |

The `FFCMConfig` struct gains additional type parameters for the
forward and backward plan types (same pattern as `FG` in step 3).

### Cold-path validation (constructor additions)

- `μ > zero(T)` else `ArgumentError("μ must be positive; got ...")`.

All step-1–3 validation is retained.

### Hot-path input (per `stokes_solve!` call)

- `config.force_grid` — populated by `spread_forces!` (step 3); the
  spread force field
  $\widetilde{\mathcal{J}}^\dagger[\mathcal{F}](\bm x_g)$.

### Hot-path output

- `config.velocity_grid` populated with $\bm u(\bm x_g)$ — the fluid
  velocity at every grid point $\bm x_g$, satisfying the discrete
  periodic Stokes equations.

### Side effects

- `config.fluid_hat` is overwritten with intermediate Fourier-domain
  data. It carries no between-call invariant.
- `config.force_grid` is **not** modified by step 4 (preserves the
  step-3 output for debuggability and future passes).

### Periodicity contract

The discrete operator is exact in the trigonometric polynomial basis
that the FFT represents: the mapping from `force_grid` to
`velocity_grid` is the discrete triply-periodic Stokes solution at the
chosen grid resolution.

### Boundary cases

- **Zero forcing.** `force_grid = 0` ⇒ `velocity_grid = 0` exactly.
- **Mean-zero forcing.** The continuous problem requires
  $\int_\Omega \widetilde{\mathcal{J}}^\dagger[\mathcal{F}]\, d^3\bm x = 0$
  for solvability; step 3's force conservation ensures this holds up
  to truncation. The kernel zeros the $\bm k = \bm 0$ mode
  unconditionally — any non-zero mean in the input is silently
  discarded by gauge-fixing (see [Mean-flow gauge fix](#mean-flow-gauge-fix)).
- **Single grid point at $\bm k = \bm 0$.** Handled by a constant-time
  guarded write at the `(1, 1, 1)` index.

## API

```julia
"""
    FFCMConfig{T}(; L, R_c, N, a = T(1), Σ_over_σ, num_grid_points, M_G, μ)

Cold-path configuration of the Fast FCM mobility operator (extended
through step 4). See `spec/spatial-hashing.md`,
`spec/particle-sorting.md`, `spec/force-spreading.md`,
`spec/stokes-solve.md`.

Keyword arguments specific to step 4:

- `μ::T` — fluid dynamic viscosity (paper eq 152, `outline.tex:152`).
  Must be positive.
"""
struct FFCMConfig{T <: AbstractFloat, FG, FH, FwdPlan, BwdPlan}
    # … existing fields from steps 1–3 …
    μ::T
    velocity_grid::FG
    fluid_hat::FH
    k_x::Vector{T}
    k_y::Vector{T}
    k_z::Vector{T}
    forward_plan::FwdPlan
    backward_plan::BwdPlan
end

"""
    stokes_solve!(config) -> config

Step 4 of the Fast FCM algorithm (Su & Keaveny 2024, §4 Step 4 = §3
Step `solve`). Apply the inverse Stokes operator `L^{-1}` to the
spread force field in `config.force_grid` and write the resulting
fluid velocity field to `config.velocity_grid`.

Operationally:
  1. Forward r2c FFT each component of `force_grid` into the
     corresponding component of `fluid_hat`.
  2. Apply `(I − k̂k̂ᵀ)/(μ k²) / M_total` per Fourier mode in place on
     `fluid_hat`; zero the `k = 0` mode.
  3. Backward c2r FFT each component of `fluid_hat` into the
     corresponding component of `velocity_grid`.

The `1/M_total` factor compensates the unnormalised FFTW round-trip.

Allocation-free and type-stable on `T <: AbstractFloat`. Reads
`config.force_grid`; writes `config.velocity_grid` and overwrites
`config.fluid_hat`.

See `spec/stokes-solve.md`.
"""
function stokes_solve!(config::FFCMConfig{T}) where {T} end
```

The underscore-prefixed
`_apply_inverse_stokes_kernel!(fx̂, fŷ, fẑ, k_x, k_y, k_z, μ,
inv_M_total)` is the function-barrier kernel: it takes naked complex
arrays and scalars so it is independently testable and benefits from
Julia's standard type-stability pattern. Mirrors the
`spread_forces!` / `_spread_forces_kernel!` split from step 3.

## Mathematical content

### Continuous problem (paper §2, eq 152)

The periodic Stokes equations with the spread force as right-hand
side:

$$
-\mu \Delta \bm{u} + \nabla p = \bm{f},
\qquad \nabla \cdot \bm{u} = 0,
\qquad \bm{x} \in \Omega .
$$

### Fourier-space inversion

Fourier-transform both equations. Writing $\hat{\bm u}$, $\hat p$, and
$\hat{\bm f}$ for the transforms:

$$
\mu k^2 \hat{\bm{u}} + i \bm{k}\, \hat p = \hat{\bm{f}},
\qquad
i \bm{k} \cdot \hat{\bm{u}} = 0 .
$$

Dotting the momentum equation with $\bm k$ and using incompressibility
gives $\hat p = -i\, (\bm k \cdot \hat{\bm f})/k^2$.
Back-substitution gives the projector form:

$$
\hat{\bm{u}}(\bm k)
= \frac{1}{\mu\, k^2}
\left(\bm I - \frac{\bm k \bm k^{\!\top}}{k^2}\right)
\hat{\bm{f}}(\bm k), \quad \bm k \ne \bm 0,
\qquad \hat{\bm u}(\bm 0) = \bm 0 .
$$

### Mean-flow gauge fix

At $\bm k = \bm 0$, the momentum equation reduces to
$\bm 0 = \hat{\bm f}(\bm 0)$. The solvability condition is therefore
that the total integrated force vanishes. When it does, the mean
velocity $\hat{\bm u}(\bm 0)$ is undetermined by the equations — no
constant pressure gradient can balance a spatially constant body
force, and the equations admit a one-parameter family of solutions
shifted by a uniform translation. The standard gauge choice is to fix
$\hat{\bm u}(\bm 0) = \bm 0$, which corresponds physically to working
in the frame where the volume-averaged fluid velocity vanishes.

In practice, step 3's force-conservation property
([force-spreading.md](force-spreading.md) test 4) ensures
$\int_\Omega \widetilde{\mathcal{J}}^\dagger[\mathcal{F}] \approx
\sum_n \bm F_n$ up to the documented truncation tolerance. The caller
is expected to supply $\sum_n \bm F_n = \bm 0$ for a physically
sensible mobility problem; if they do not, the gauge fix silently
discards the mean. The non-zero mean of the spread field that arises
from step-3 truncation is similarly absorbed by the gauge fix and
does not affect the computed velocity field at any non-zero Fourier
mode.

### Discrete FFT and normalisation

FFTW's r2c forward + c2r backward composition is unnormalised: a
round-trip multiplies by $M_\text{total} = M_x M_y M_z$. The
$1/M_\text{total}$ correction is folded into the per-mode scalar:

$$
\hat{\bm u}(\bm k) \leftarrow
\frac{1}{\mu\, k^2\, M_\text{total}}
\left(\bm I - \frac{\bm k \bm k^{\!\top}}{k^2}\right) \hat{\bm f}(\bm k).
$$

### Wavenumber layout

The r2c FFT halves the leading dimension: `fluid_hat` has shape
`(M_x ÷ 2 + 1, M_y, M_z)`. Per axis:

- $k_x[i] = (2\pi / L_x)\, (i - 1)$ for $i \in \{1, \ldots, M_x/2 + 1\}$
  (non-negative half-axis).
- $k_y[j] = (2\pi / L_y) \cdot n_y(j)$ with
  $n_y(j) = j - 1$ for $j \le M_y/2 + 1$, $n_y(j) = j - 1 - M_y$
  otherwise (FFTW wrap-around).
- $k_z[\ell]$ analogous to $k_y$.

### Per-mode projection kernel (closed form for the inner loop)

For one Fourier grid point `(i, j, ℓ)` with
`k = (k_x[i], k_y[j], k_z[ℓ])`:

```
k²  = k_x[i]² + k_y[j]² + k_z[ℓ]²
α   = inv_M_total / (μ · k²)
k·f̂ = k_x[i] · fx̂ + k_y[j] · fŷ + k_z[ℓ] · fẑ
c   = (k·f̂) / k²
fx̂ ← α · (fx̂ − k_x[i] · c)
fŷ ← α · (fŷ − k_y[j] · c)
fẑ ← α · (fẑ − k_z[ℓ] · c)
```

At `(i, j, ℓ) = (1, 1, 1)` (the $\bm k = \bm 0$ mode), all three
components are set to `Complex{T}(0)` instead, skipping the projection
to avoid producing `NaN` from a `1/0`.

## Cold-path vs hot-path

| Phase | Allocations | Functions |
|---|---|---|
| Cold | OK | `FFCMConfig` constructor: validate `μ > 0`; build `velocity_grid`, `fluid_hat`, `k_x`, `k_y`, `k_z`, the forward r2c FFT plan, and the backward c2r FFT plan. |
| Hot  | `@ballocated == 0` | `stokes_solve!(config)` — three forward FFTs, the projection kernel, and three backward FFTs. |

The hot path is allocation-free and type-stable on `T <: AbstractFloat`.

## Performance notes

- **Storage.** `velocity_grid` mirrors `force_grid` SoA-wise — three
  backing `Array{T, 3}` wrapped in a `StructArray{SVector{3, T}, 3}`.
  Step 5 (interpolation) will read per-grid-point `SVector{3, T}`
  velocities, exactly mirroring step 3's per-grid-point force read.
- **Single forward / backward plan, applied three times.** The three
  components are independent real or complex arrays of the same shape
  and type, so one `plan_rfft` and one `plan_brfft` suffice. Calling
  `mul!(complex_out, plan, real_in)` per component is allocation-free
  once the plan is constructed.
- **Plan flags.** Cold-path FFTW plan flags are deferred — start with
  `FFTW.ESTIMATE` (no plan-time overhead, modest runtime cost) and
  upgrade to `FFTW.MEASURE` if benchmarks show plan execution to be the
  bottleneck. Plan-flag choice is a benchmark concern, not a
  correctness one.
- **`fluid_hat` is reused for $\hat f$ and $\hat u$.** The
  Fourier-space buffer transitions from "Fourier-domain force" to
  "Fourier-domain velocity" in place during the kernel. Documenting
  the aliasing keeps the cold-path field set at one rather than two,
  with no semantic loss — the projection is per-Fourier-point local,
  so read/write at the same index is race-free.
- **Wavenumbers precomputed once.** `k_x`, `k_y`, `k_z` are 1-D
  `Vector{T}` at cold path. The hot-path kernel composes
  $k^2 = k_x[i]^2 + k_y[j]^2 + k_z[\ell]^2$ at each Fourier point
  with two adds and three squares — cheaper than a 3-D
  wavenumber-magnitude buffer that would not fit in L1 for typical
  grids.
- **Inner loop walks stride-1 along the r2c half-axis.** Column-major
  storage of the complex backing arrays means `kz` is the outer loop,
  `ky` the middle, `kx` the inner. The projection is purely local in
  `(i, j, ℓ)`.
- **`k = 0` mode handling.** A single conditional guarded by
  `(i, j, ℓ) == (1, 1, 1)` writes `Complex{T}(0)` to the three
  components. The fix is constant-time and isolated to one Fourier
  point.
- **Plan thread count.** Defaults to whatever FFTW's process-global
  `FFTW.set_num_threads` is set to. Not enforced at cold path. The
  parallelisation strategy for step 4 sits under the same PLAN.md
  "CPU optimisation" backlog item as the other steps.

## Diffs from cuFCM

Audited cuFCM files: `cuFCM/src/CUFCM_FCM.cu:186-247`
(`cufcm_flow_solve`, the Fourier-space inverse Stokes kernel) and
`cuFCM/src/CUFCM_SOLVER.cu` for orchestration (plan construction at
`:230-238`, real and complex buffer setup at `:200-212`, the
`fft_solve` method at `:601-642` that sequences forward FFT →
`cufcm_flow_solve` → backward FFT).

| Facet | cuFCM | This package | Decision |
|---|---|---|---|
| Real-space buffers | `hx, hy, hz` reused as both input (force) and output (velocity) of the FFT round-trip. | Separate `force_grid` and `velocity_grid`. | **Diverge.** Debuggability and step 3 ↔ step 4 decoupling outweigh the cold-path memory cost of three extra `Array{T, 3}`. |
| Fourier-space buffers | Separate `fk_x, fk_y, fk_z` (force, forward-FFT output) and `uk_x, uk_y, uk_z` (velocity, backward-FFT input). The projection reads `fk_*` and writes `uk_*`. | Single `fluid_hat` overwritten in place by the projection. | **Diverge.** The projection is per-Fourier-point local; single buffer is race-free in the serial loop and any future per-grid-point parallelisation. Saves three `Array{Complex{T}, 3}` of cold-path memory. cuFCM's split is plausibly defensive (GPU input/output discipline) rather than required. |
| Number of FFT plans | One r2c (`plan`) and one c2r (`iplan`), each applied three times per call (once per component). | Identical: one `forward_plan`, one `backward_plan`, each applied three times per call. | **Adopt.** Independently natural — the three components share shape and type. |
| FFT plan flags / planner mode | cuFFT has no flag analogue at `cufftPlan3d` time. | `FFTW.ESTIMATE` for the MVP; `MEASURE` if benchmarks demand. | **Out of scope.** No comparable knob. |
| Per-mode wavenumber | Computed inline in the kernel: `q_i = (ind_i ≤ M_i/2 ? ind_i : ind_i − M_i) · 2π/L_i`. | Precomputed `k_x, k_y, k_z::Vector{T}` at cold path. | **Diverge.** Inline compute is a GPU choice (every thread runs identical arithmetic, no branch divergence). On CPU, precomputing saves three branches and six multiplies per Fourier point at trivial cold-path memory cost. |
| $\bm k = \bm 0$ mode handling | Compute-then-fix: `qq_inv = 1/0 = ∞`, `kdotf = 0·∞ = NaN`, `norm·(...) = NaN`, then unconditionally overwrites `uk_*[0]` with zero. | Single explicit write of `Complex{T}(0)` to all three components at `(1, 1, 1)`, gated by a branch. No `NaN` intermediates. | **Diverge.** cuFCM's compute-then-fix avoids intra-warp branch divergence; on CPU a single branch is strictly faster and prevents the `NaN` from polluting any future numerical debugging. |
| Normalisation point | Folded into the per-mode scalar: `norm = qq_inv / grid_size`. | Same: `α = inv_M_total / (μ · k²)`, applied to the projected $\hat f$. | **Adopt.** Independently natural. |
| Viscosity | Implicit $\mu = 1$ (non-dimensional unit-viscosity convention). | Explicit `μ::T` on `FFCMConfig`, validated `> 0`. | **Diverge.** Keep the explicit viscosity. Cost: one cold-path division (`1/μ` precomputed) and one extra multiply per Fourier point. Benefit: downstream users (resistance solvers, suspension dynamics) plug in their physical $\mu$ without having to non-dimensionalise. |
| Component plumbing | Six separately-named arrays (`fk_x, fk_y, fk_z, uk_x, uk_y, uk_z`) passed by pointer to the kernel. | `(fx̂, fŷ, fẑ) = StructArrays.components(fluid_hat)` destructured once per call. | **Diverge.** Type-level SoA discipline via `StructArray`. Internally identical memory layout. |
| Loop / parallelism | One CUDA thread per Fourier grid point; linear index `i` unpacks with `indi` fastest (matches our column-major `(i, j, ℓ)` layout). | Single threaded triple loop, `kz` outer, `kx` inner (stride-1 along the r2c half-axis). | **Keep** the serial CPU loop for the MVP. Future CPU threading is straightforward and lives under the existing PLAN.md "CPU optimisation" backlog. |
| FFT call sequencing | `plan` (r2c) for each of `hx, hy, hz`; then `cufcm_flow_solve`; then `iplan` (c2r) for each of `uk_x, uk_y, uk_z`. | Identical: three `mul!(fluid_hat.x̂, forward_plan, force_grid.x)`, one `_apply_inverse_stokes_kernel!`, three `mul!(velocity_grid.x, backward_plan, fluid_hat.x̂)`. | **Adopt.** Same logical sequence. |
| Error handling | Each `cufft*Exec*` checks `!= CUFFT_SUCCESS`, prints, and returns. | No per-call check. FFTW.jl `mul!` is infallible once the plan is constructed. | **Keep.** Validate cold, trust hot. |
| Stored fft grid size | `fft_grid_size = (nx/2 + 1) · ny · nz` stored on the solver struct. | Derived from `num_grid_points` at cold path. | **Keep.** One arithmetic op, no benefit to caching. |

The behavioural differences with implementation consequence are the
explicit-`μ` parameterisation (the only diff that touches the
arithmetic), the precomputed wavenumber vectors, the single-buffer
`fluid_hat`, and the clean $\bm k = \bm 0$ early-exit. The remaining
diffs are either independent (same conclusion reached from different
starting points) or CPU-vs-GPU code-organisation choices with no
algorithmic content.

## GPU revisit notes

The CPU-driven choices below will need re-examination when the CUDA
backend is designed (`PLAN.md > Backlog`: *CUDA.jl implementation*).
The CPU code is correct in isolation; transcribing it line-for-line
to CUDA would forfeit the GPU performance characteristics the cuFCM
reference is known to deliver.

- **$\bm k = \bm 0$ early-exit (branch).** CPU benefits from skipping
  the projection at one Fourier point; GPU pays for the branch via
  intra-warp divergence. cuFCM's compute-then-fix (let every thread
  run the same arithmetic, then unconditionally zero the result at
  `i = 0` at the end) is the GPU-natural pattern. On revisit, either
  port cuFCM's pattern or check whether modern CUDA architectures
  handle a single divergent thread cheaply enough to keep the branch.
- **Precomputed wavenumber vectors.** On CPU the lookup is cache-warm
  and saves three branches per Fourier point. On GPU the lookup
  costs a global-memory read while every thread already computes the
  index arithmetic from its block / thread IDs; cuFCM's inline
  compute is almost certainly faster on GPU. Revisit: probably
  inline in the CUDA kernel.
- **Serial triple loop.** Will be replaced by one CUDA thread per
  Fourier grid point (matching `cufcm_flow_solve`'s linear-index
  decomposition). Trivial port.
- **Single-buffer `fluid_hat`.** Race-free in any per-grid-point
  parallel scheme (read and write at the same index per thread).
  cuFCM's split (`fk_* / uk_*`) is plausibly defensive rather than
  required. Revisit: keep the single buffer unless profiling shows a
  cache-line / coalescing reason to split.

## Verification

End-to-end correctness for this step is established by the test
suite. Test files follow the flat-`test/` layout and the
domain-language naming convention.

`test/test_stokes_solve.jl`:

1. **Cold-path validation.** `μ ≤ 0` raises `ArgumentError`. With a
   valid construction, `velocity_grid` has shape `(M_x, M_y, M_z)`
   and `StructArrays.components(velocity_grid)` is a 3-tuple of
   `Array{T, 3}` of that shape. `fluid_hat` has shape
   `(M_x ÷ 2 + 1, M_y, M_z)` with components `Array{Complex{T}, 3}`.
   `k_x`, `k_y`, `k_z` have the expected lengths and a sampled index
   matches the closed-form layout above.
2. **Zero forcing ⇒ zero velocity.** `force_grid` set to zero ⇒
   `velocity_grid` is exactly zero after `stokes_solve!`.
3. **Mean velocity is zero.** For arbitrary mean-zero forcing
   (constructed by spreading several particles with
   $\sum_n \bm F_n = 0$), `sum(velocity_grid_component) ≈ 0` per
   component to `sqrt(eps(T))`.
4. **Discrete incompressibility ($\nabla \cdot \bm u = 0$).** The
   exact identity the projector enforces is
   $\bm k \cdot \hat{\bm u}(\bm k) = 0$ at every Fourier mode. Test
   by re-forward-transforming `velocity_grid` and asserting
   $\lvert \bm k \cdot \hat{\bm u} \rvert \le \mathrm{sqrt}(\mathrm{eps}(T))
   \cdot \lVert \hat{\bm u} \rVert$ per mode. A centred
   finite-difference divergence on the real-space grid would only be
   $O(\Delta x^2)$ — not the invariant the projector enforces, so
   not the right test.
5. **Single Fourier mode — analytical projection.** Set `force_grid`
   to a pure mode along $\hat x$ with forcing direction $\hat y$:
   $\bm f(\bm x) = \sin(2\pi m x / L_x)\, \hat{\bm e}_y$ for integer
   $m \in (0, M_x/2)$. Since $\bm k$ is along $\hat x$ and $\bm f$
   along $\hat y$, the projector acts as the identity and
   $\hat{\bm u}(\bm k) = \hat{\bm f}(\bm k) / (\mu k_x^2)$ at the two
   conjugate-symmetric nonzero modes, zero elsewhere. Compare the
   resulting `velocity_grid` to the closed form at `sqrt(eps(T))`.
6. **Linearity in `force_grid`.** `stokes_solve!` with input
   $\alpha \bm f_1 + \beta \bm f_2$ equals
   $\alpha \cdot (\text{solve } \bm f_1)
   + \beta \cdot (\text{solve } \bm f_2)$.
7. **Periodicity.** Translate `force_grid` by one grid step along
   axis $i$ (cyclic shift) ⇒ `velocity_grid` is the cyclic shift of
   the un-translated `velocity_grid` by the same amount.
8. **Reflection symmetry.** Reflect `force_grid` along an axis ⇒
   `velocity_grid` reflects analogously (in-plane components
   reflect; the out-of-plane component flips sign per the
   projector parity). The test uses an odd grid dimension on every
   axis to avoid the Nyquist artefact described below.

   **Nyquist artefact (odd vs even grids).** For even `M_i`, the
   Nyquist mode `k_i = π/Δx_i` is its own conjugate-symmetric partner
   on the r2c discrete grid — the sign convention `k_y[Nyq] = +π/Δy`
   breaks exact reflection equivariance of the projector's
   off-diagonal entries at that single mode. This is a well-known
   artefact of the r2c FFT layout, not a defect of the projector
   arithmetic: the cycle-5 single-Fourier-mode test pins the
   projector at non-Nyquist modes to `sqrt(eps(T))`. Choosing odd
   `M_i` removes the Nyquist mode entirely; for even `M_i` callers
   should expect a small discrepancy at the Nyquist contribution.
9. **Viscosity scaling.** `velocity_grid` from `μ = μ₀` and
   `velocity_grid` from `μ = 2μ₀` (same forcing) differ by exactly a
   factor of 2 to `sqrt(eps(T))`.
10. **End-to-end spread + solve at point-forcing limit.** One
    particle at a chosen position, `Σ/σ = 1` (standard-FCM
    degenerate limit, so the input is the plain Gaussian-spread
    point force). Compare `velocity_grid` at grid points well
    outside the kernel support to the regularised Stokeslet
    $\bm S(\bm x_g - \bm Y_n; \sigma\sqrt 2)$ from paper eq 205–207,
    in its periodised form (sum over a few image cells of magnitude
    $L$). Tolerance and image-sum cutoff are documented inline. This
    is the one test where the result depends on every step from 1
    through 4 composed.

`test/test_stokes_solve_inferred.jl` — `@inferred` type stability for
`stokes_solve!` and `_apply_inverse_stokes_kernel!`,
`Float32`/`Float64`.

`test/test_stokes_solve_allocations.jl` — `@ballocated == 0` for both
the wrapper and the kernel.

`test/test_jet.jl` — `JET.@test_call stokes_solve!` walks the full
call graph for inference health.

`test/test_aqua.jl` — `Aqua.test_all` covers package hygiene; the
new FFTW dependency must surface cleanly.

`test/test_fcm_grid.jl` — pin the new cold-path derived fields
(`velocity_grid` shape; `fluid_hat` shape and complex component type;
wavenumber vector lengths and closed-form values at a sampled index).

Tolerances: tests 2–9 use `sqrt(eps(T))` (the discrete identities are
exact to round-off). Test 10 uses the paper truncation tolerance with
documented periodic-image cutoff.
