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
for $\boldsymbol{k} \neq 0$ and $\hat{\boldsymbol{u}}(\boldsymbol{0}) = \boldsymbol{0}$, where
$\boldsymbol{k}$ is the wavenumber vector of the 3D Fourier transform.

The spectral method is then to apply the FFT to the spread forces sampled on the grid to obtain $\hat{\boldsymbol{f}}$, solve for $\hat{\boldsymbol{u}}$, and apply the inverse FFT to recover $\boldsymbol{u}$ sampled on the grid.

In the code, we define
- `μ` $= \mu$,
- `force_density[i_x, i_y, i_z]` $= \boldsymbol{f}\bigl((i_x - 1)h, (i_y - 1)h, (i_z - 1)h\bigr)$,
- `fluid_velocity[i_x, i_y, i_z]` $= \boldsymbol{u}\bigl((i_x - 1)h, (i_y - 1)h, (i_z - 1)h\bigr)$,
- `fluid_hat[i_x, i_y, i_z]` $= \hat{\boldsymbol{f}}\bigl(k_x[i_x], k_y[i_y], k_z[i_z]\bigr)$ initially, rewritten to $\hat{\boldsymbol{u}}$ after the solve.

## Method

### Fourier-space inversion

Taking the Fourier transform of the momentum and continuity equations turns the
differential operators into multiplications. Writing $\hat{\boldsymbol{u}}$, $\hat p$, and
$\hat{\boldsymbol{f}}$ for the transforms and $k = \lvert\boldsymbol{k}\rvert$,
$$
\mu k^2 \hat{\boldsymbol{u}} + i \boldsymbol{k} \hat p = \hat{\boldsymbol{f}},
\qquad
i \boldsymbol{k} \cdot \hat{\boldsymbol{u}} = 0 .
$$
Dotting the momentum equation with $\boldsymbol{k}$ and using incompressibility
($\boldsymbol{k} \cdot \hat{\boldsymbol{u}} = 0$) eliminates the velocity term and gives
the pressure,
$$
i k^2 \hat p = \boldsymbol{k} \cdot \hat{\boldsymbol{f}}
\quad\Longrightarrow\quad
\hat p = -i \frac{\boldsymbol{k} \cdot \hat{\boldsymbol{f}}}{k^2}.
$$
Substituting back,
$$
\mu k^2 \hat{\boldsymbol{u}}
= \hat{\boldsymbol{f}} - i\boldsymbol{k}\hat p
= \hat{\boldsymbol{f}} - \frac{\boldsymbol{k}(\boldsymbol{k} \cdot \hat{\boldsymbol{f}})}{k^2},
$$
which is the projector form quoted in the Summary,
$$
\hat{\boldsymbol{u}}(\boldsymbol{k})
= \frac{1}{\mu k^2}\left(\boldsymbol{I} - \frac{\boldsymbol{k}\otimes\boldsymbol{k}}{k^2}\right)\hat{\boldsymbol{f}}(\boldsymbol{k}),
\qquad \boldsymbol{k} \neq \boldsymbol{0}.
$$

### The mean-flow gauge fix

At $\boldsymbol{k} = \boldsymbol{0}$ the momentum equation reduces to
$\boldsymbol{0} = \hat{\boldsymbol{f}}(\boldsymbol{0})$, so the solvability condition is
that the total integrated force vanishes. When it does, the mean velocity
$\hat{\boldsymbol{u}}(\boldsymbol{0})$ is undetermined by the equations — no constant
pressure gradient can balance a spatially constant body force, and the solution is fixed
only up to a uniform translation. The standard gauge choice sets
$\hat{\boldsymbol{u}}(\boldsymbol{0}) = \boldsymbol{0}$, which is the frame where the
volume-averaged fluid velocity vanishes. Step 3's force conservation
([force-spreading.md](force-spreading.md)) makes the spread force sum to
$\sum_n \boldsymbol{F}_n$ up to truncation, so for a physical mobility problem with
$\sum_n \boldsymbol{F}_n = \boldsymbol{0}$ the discarded mean is (up to truncation) zero;
any residual mean is absorbed by the gauge fix and does not affect any non-zero mode.

### Discrete transform and normalisation

The implementation uses FFTW's real-to-complex (r2c) forward transform and
complex-to-real (c2r) backward transform. Their composition is unnormalised: a round-trip
multiplies by the total grid-point count $M = M_x M_y M_z$. That factor is folded into the
per-mode scalar, so the projection applied in place is
$$
\hat{\boldsymbol{u}}(\boldsymbol{k}) \leftarrow
\frac{1}{\mu k^2 M}\left(\hat{\boldsymbol{f}}(\boldsymbol{k}) - \frac{\boldsymbol{k}(\boldsymbol{k}\cdot\hat{\boldsymbol{f}}(\boldsymbol{k}))}{k^2}\right),
\qquad
\hat{\boldsymbol{u}}(\boldsymbol{0}) = \boldsymbol{0}.
$$

### Wavenumber layout

The r2c transform halves the leading dimension, so `fluid_hat` has shape
$(M_x/2 + 1, M_y, M_z)$. The per-axis wavenumbers are
$$
k_x[i_x] = \frac{2\pi}{L_x}(i_x - 1), \quad i_x \in \{1, \dots, M_x/2 + 1\} \quad (\text{the non-negative half-axis}),
$$
$$
k_y[i_y] = \frac{2\pi}{L_y} n_y(i_y), \quad n_y(i_y) = \begin{cases} i_y - 1, & i_y \leq M_y/2 + 1, \\ i_y - 1 - M_y, & \text{otherwise}, \end{cases}
$$
and $k_z$ is analogous to $k_y$. The conditional is FFTW's wrap-around convention, which
places the negative frequencies in the upper half of each full axis. The zero mode sits at
the leading index $i_x = i_y = i_z = 1$, where the kernel applies the gauge fix.

## Contract

### Cold-path input (new in this step)

- `μ` $= \mu$ (`T`) — the fluid dynamic viscosity, required on `FFCMConfig`. Must satisfy
  $\mu > 0$. No default; viscosity is a physical quantity and should be an explicit caller
  decision.
- `fft_planning` (`Symbol`, default `:measure`) — the FFT planner effort, one of
  `:estimate`, `:measure`, or `:patient` (mapped to the corresponding FFTW planner
  flags). Planner effort trades construction time for transform time: `:estimate` plans
  immediately from heuristics; `:measure` and `:patient` time candidate algorithms on
  the actual grid at construction (seconds to minutes at large grids) and can produce
  faster plans for the many `stokes_solve!` calls of a long resistance solve. The
  solution `fluid_velocity` is independent of the planner effort to round-off. Any other
  value raises `ArgumentError`.
- `fft_threads` (`Integer`, default `1`) — the execution thread count baked into the two
  FFT plans at construction. Threaded plans execute as Julia tasks, which allocate per
  call: the hot-path allocation-free guarantee is scoped to `fft_threads = 1`. Must be
  at least 1, else `ArgumentError`. The solution matches the single-threaded result to
  round-off (the task partition changes summation order only).

The cold-path inputs from steps 1–3 (`L`, `R_c`, `N`, `a`, `Σ_over_σ`, `num_grid_points`,
`M_G`) are unchanged.

### Cold-path derived state (added to `FFCMConfig`)

| Field | Type | Definition |
|---|---|---|
| `μ` | `T` | The user input. |
| `fluid_velocity` | `StructArray{SVector{3, T}, 3, …}` | Shape $(M_x, M_y, M_z)$, the same struct-of-arrays layout as `force_density`. Holds $\boldsymbol{u}(\boldsymbol{x}_g)$ after the solve. |
| `fluid_hat` | `StructArray{SVector{3, Complex{T}}, 3, …}` | Shape $(M_x/2 + 1, M_y, M_z)$, three backing `Array{Complex{T}, 3}`. Holds $\hat{\boldsymbol{f}}$ on entry to the projection and $\hat{\boldsymbol{u}}$ on exit. |
| `k_x` | `Vector{T}` | Length $M_x/2 + 1$; the wavenumbers of the r2c half-axis. |
| `k_y`, `k_z` | `Vector{T}` | Lengths $M_y$, $M_z$; the full-axis wavenumbers with FFTW wrap-around. |
| `forward_fourier_transform` | FFTW r2c plan | Built once; applies to any `Array{T, 3}` of shape $(M_x, M_y, M_z)$. |
| `inverse_fourier_transform` | FFTW c2r plan | Built once; applies to any `Array{Complex{T}, 3}` of shape $(M_x/2 + 1, M_y, M_z)$. |

`FFCMConfig` carries extra type parameters for the two plan types.

Cold-path validation (constructor addition): $\mu > 0$,
`fft_planning ∈ (:estimate, :measure, :patient)`, and `fft_threads ≥ 1`, else
`ArgumentError`. All step-1–3 validation is retained.

### Hot-path input (per `stokes_solve!` call)

- `config.force_density` — the spread force field
  $\tilde{\mathcal{J}}^\dagger[\mathcal{F}](\boldsymbol{x}_g)$ from `spread_forces!`.

### Hot-path output

- `config.fluid_velocity` — the fluid velocity $\boldsymbol{u}(\boldsymbol{x}_g)$ at every
  grid point, satisfying the discrete periodic Stokes equations.

### Side effects

- `config.fluid_hat` is overwritten with intermediate Fourier-domain data; it carries no
  between-call invariant.
- `config.force_density` is **not** modified (the step-3 output is preserved for
  debuggability).

### Periodicity contract

The discrete operator is exact in the trigonometric-polynomial basis the FFT represents:
the map from `force_density` to `fluid_velocity` is the discrete triply-periodic Stokes
solution at the chosen grid resolution.

### Boundary cases

- **Zero forcing.** `force_density` $= 0 \Rightarrow$ `fluid_velocity` $= 0$ exactly.
- **Mean-zero forcing.** The kernel zeros the $\boldsymbol{k} = \boldsymbol{0}$ mode
  unconditionally; any non-zero mean in the input is discarded by the gauge fix.
- **The $\boldsymbol{k} = \boldsymbol{0}$ point.** Handled by a constant-time guarded write
  of zero at the leading index, avoiding a $1/0$.

## Implementation

`stokes_solve!(config)` is the hot-path entry. It reads `config.force_density`, writes
`config.fluid_velocity`, and overwrites `config.fluid_hat`, returning `config`. It runs in
three stages: a forward r2c FFT of each of the three force components into `fluid_hat`; the
per-mode projection applied in place on `fluid_hat`; and a backward c2r FFT of each
component into `fluid_velocity`. It is allocation-free and type-stable.

The projection is the function-barrier kernel
`_apply_inverse_stokes_kernel!(fx̂, fŷ, fẑ, k_x, k_y, k_z, μ, inv_M)`, which takes
the three naked complex component arrays and the wavenumber vectors so it is type-stable
and independently testable. It walks the Fourier grid, computes $k^2$ from the precomputed
wavenumbers, applies the normalised projector above, and writes zero at the
$\boldsymbol{k} = \boldsymbol{0}$ mode.

The three force components share shape and type, so a single `forward_fourier_transform` and a single
`inverse_fourier_transform` suffice, each applied three times via `mul!`. `fluid_hat` is reused for
both $\hat{\boldsymbol{f}}$ and $\hat{\boldsymbol{u}}$: the projection is local in each
Fourier point, so the in-place read-then-write at one index is race-free. The grids, the
wavenumber vectors, and the two plans are built once by the `FFCMConfig` constructor.
The constructor builds the plans on the freshly allocated grid buffers **before** zeroing
them: FFTW's measuring planner levels evaluate candidate algorithms by executing them on
the input array, overwriting its contents, so the plan-then-zero order guarantees that
"all grid buffers are zero after construction" holds at every planner effort.

| Phase | Allocations | Functions |
|---|---|---|
| Cold | OK | `FFCMConfig` constructor: validate $\mu > 0$; build `fluid_velocity`, `fluid_hat`, `k_x`/`k_y`/`k_z`, and the two FFT plans. |
| Hot  | `@ballocated == 0` | `stokes_solve!(config)` — three forward FFTs, the projection kernel, three backward FFTs. |

The hot path is allocation-free and type-stable on `T <: AbstractFloat`.

## Performance notes

- **Storage.** `fluid_velocity` mirrors `force_density` (three backing `Array{T, 3}` wrapped
  in a `StructArray`), so step 5 reads a per-grid-point `SVector{3, T}` velocity exactly
  as step 3 wrote a force.
- **One plan, applied three times.** The three components are independent arrays of the
  same shape, so one `plan_rfft` and one `plan_brfft` cover them; `mul!(out, plan, in)`
  per component is allocation-free once the plan exists.
- **Planner effort.** The default is `:measure`: on the benchmarked 64³/128³ grids the
  measured plans run `stokes_solve!` 9–22 % faster than `:estimate` at both precisions,
  for a one-time planning cost of ≈ 0.3 s (64³) to ≈ 1.3 s (128³) that amortises over
  the repeated `stokes_solve!` calls of a resistance solve (see the benchmark suite for
  current numbers on a given machine). Choose `:estimate` when construction latency
  matters more than per-call speed — small grids, one-shot evaluations. Users can cache
  planner results across sessions with FFTW's wisdom mechanism — `FFTW.import_wisdom`
  before constructing the config, `FFTW.export_wisdom` after — with no FFCM
  involvement.
- **FFT threading.** `fft_threads > 1` bakes multi-threaded execution into both plans;
  with the FFTW provider the transform body runs as spawned Julia tasks, so each
  `stokes_solve!` call allocates task state — the allocation-free hot-path guarantee is
  scoped to `fft_threads = 1`. The thread count is per-plan state: FFTW.jl
  saves and restores the planner's thread setting internally, so construction leaves no
  global FFTW state behind. With the MKL provider the threading is MKL-internal and the
  allocation caveat differs. Whether threading pays is grid-size- and machine-dependent;
  the `fft-threads` benchmark group measures it on the host at hand.
- **`fluid_hat` reused for $\hat{\boldsymbol{f}}$ and $\hat{\boldsymbol{u}}$.** One
  Fourier buffer rather than two; the per-point-local projection makes the in-place
  transition safe.
- **Wavenumbers precomputed once.** The hot-path kernel composes
  $k^2 = k_x[i]^2 + k_y[j]^2 + k_z[\ell]^2$ with two adds and three squares per Fourier
  point — cheaper than a 3-D wavenumber-magnitude buffer that would not fit in L1.
- **Stride-1 inner loop.** Column-major complex arrays make $k_z$ the outer loop, $k_x$
  the inner; the projection is purely local in the Fourier index.
- **The $\boldsymbol{k} = \boldsymbol{0}$ mode** is a single constant-time guarded write of
  zero.
- **GPU revisit.** The CPU-favourable choices here will need re-examination for the future
  GPU backend: the $\boldsymbol{k} = \boldsymbol{0}$ branch (a divergent thread on the GPU,
  where a compute-then-zero pattern is more natural), the precomputed wavenumber vectors
  (a global-memory read on the GPU versus index arithmetic each thread already has), and
  the single in-place `fluid_hat` buffer (race-free per Fourier point, so it should
  transcribe directly).

## Verification

End-to-end correctness for this step is established by the test suite.

`test/test_stokes_solve.jl`:

1. **Cold-path validation.** $\mu \leq 0$ raises `ArgumentError`. With a valid
   construction, `fluid_velocity` has shape $(M_x, M_y, M_z)$ with `Array{T, 3}`
   components; `fluid_hat` has shape $(M_x/2 + 1, M_y, M_z)$ with `Array{Complex{T}, 3}`
   components; the wavenumber vectors have the expected lengths and a sampled index matches
   the closed-form layout.
2. **Zero forcing ⇒ zero velocity**, exactly.
3. **Mean velocity is zero.** For mean-zero forcing (several particles with
   $\sum_n \boldsymbol{F}_n = 0$), each velocity component sums to $\approx 0$ to
   `sqrt(eps(T))`.
4. **Discrete incompressibility.** The projector enforces
   $\boldsymbol{k} \cdot \hat{\boldsymbol{u}}(\boldsymbol{k}) = 0$ at every mode; re-forward
   transform `fluid_velocity` and assert
   $\lvert \boldsymbol{k} \cdot \hat{\boldsymbol{u}} \rvert \leq \texttt{sqrt(eps(T))}\cdot\lVert\hat{\boldsymbol{u}}\rVert$ per mode.
5. **Single Fourier mode — analytical projection.** Set `force_density` to a pure mode along
   $\hat{x}$ with forcing along $\hat{y}$: $\boldsymbol{f}(\boldsymbol{x}) = \sin(2\pi m x / L_x)\hat{\boldsymbol{e}}_y$
   for integer $m \in (0, M_x/2)$. The projector acts as the identity, so
   $\hat{\boldsymbol{u}} = \hat{\boldsymbol{f}} / (\mu k_x^2)$ at the two conjugate-symmetric
   nonzero modes; compare `fluid_velocity` to the closed form at `sqrt(eps(T))`.
6. **Linearity in `force_density`.**
7. **Periodicity.** A cyclic shift of `force_density` by one grid step on axis $i$ gives the
   same cyclic shift of `fluid_velocity`.
8. **Reflection symmetry.** Reflecting `force_density` along an axis reflects `fluid_velocity`
   analogously (in-plane components reflect; the out-of-plane component flips sign per the
   projector parity). The test uses odd grid dimensions to avoid the Nyquist artefact:
   for even $M_i$ the Nyquist mode $k_i = \pi/h$ is its own conjugate-symmetric partner
   on the r2c grid, which breaks exact reflection equivariance of the projector's
   off-diagonal at that single mode — a known r2c-layout artefact, not a projector defect.
9. **Viscosity scaling.** `fluid_velocity` from $\mu = \mu_0$ and from $\mu = 2\mu_0$ (same
   forcing) differ by exactly a factor of 2 to `sqrt(eps(T))`.
10. **End-to-end spread + solve at the point-forcing limit.** One particle, $\Sigma/\sigma = 1$,
    compare `fluid_velocity` at grid points well outside the kernel support to the
    regularised Stokeslet $\boldsymbol{S}(\boldsymbol{x}_g - \boldsymbol{Y}_n; \sigma\sqrt 2)$
    (§2, equation (14); Fourier form in §3, equations (32)–(33), and Appendix A) in its periodised form (a few image cells).
    Tolerance and image cutoff documented inline. This is the one test exercising steps 1–4
    composed.

`test/test_stokes_solve_inferred.jl` — `@inferred` for `stokes_solve!` and the kernel,
`Float32`/`Float64`. `test/test_stokes_solve_allocations.jl` — `@ballocated == 0`.
`test/test_jet.jl` — `JET.@test_call stokes_solve!`. `test/test_aqua.jl` — package hygiene
(the FFTW dependency must surface cleanly). `test/test_fcm_grid.jl` — pins the new
cold-path derived fields.

`test/accuracy/test_fft_planning.jl` — the single-sphere periodic self-mobility is
independent of the planner effort: `:estimate` and `:measure` configs both reproduce the
lattice-sum reference, and their `mobility!` results agree to `sqrt(eps(T))` on the
**first** call after construction (which would catch planner scribble surviving in a grid
buffer); an unknown planner effort raises `ArgumentError`.

`test/accuracy/test_fftw_threading.jl` — a threaded-plan config reproduces the
single-thread reference: the same spread forces give the same `fluid_velocity` and the
same `mobility!` velocities to `sqrt(eps(T))`; a non-positive thread count raises
`ArgumentError`. `test/api/test_stokes_solve_api.jl` adds `@inferred stokes_solve!` on a
threaded config (no allocation assertion: threaded execution allocates by design).

Tolerances: tests 2–9 use `sqrt(eps(T))` (the discrete identities are exact to round-off);
test 10 uses the paper truncation tolerance with a documented periodic-image cutoff.

## Differences from cuFCM

> Comparison against the C++/CUDA reference implementation, kept for validation during
> development and removed once the port is complete.

Audited: `cuFCM/src/CUFCM_FCM.cu` (`cufcm_flow_solve`, the Fourier-space inverse Stokes
kernel) and `cuFCM/src/CUFCM_SOLVER.cu` (plan construction, buffer setup, and the
`fft_solve` sequencing forward FFT → `cufcm_flow_solve` → backward FFT).

| Facet | cuFCM | This package | Decision |
|---|---|---|---|
| Real-space buffers | `hx, hy, hz` reused as both FFT input (force) and output (velocity). | Separate `force_density` and `fluid_velocity`. | **Diverge** — debuggability and step 3 ↔ 4 decoupling outweigh three extra `Array{T, 3}`. |
| Fourier-space buffers | separate `fk_*` (force) and `uk_*` (velocity). | a single `fluid_hat` overwritten in place. | **Diverge** — the projection is per-point-local, so one buffer is race-free; saves three complex arrays. |
| Number of FFT plans | one r2c and one c2r, each applied three times. | identical. | **Adopt** — independently natural. |
| Per-mode wavenumber | computed inline in the kernel. | precomputed `k_x/k_y/k_z`. | **Diverge** — inline compute is a GPU choice; on CPU precomputing saves branches and multiplies per Fourier point. |
| $\boldsymbol{k} = \boldsymbol{0}$ handling | compute-then-fix: produce `NaN`, then overwrite with zero. | an explicit guarded write of zero, no `NaN` intermediates. | **Diverge** — on CPU a single branch is faster and avoids polluting debugging with `NaN`. |
| Normalisation point | folded into the per-mode scalar. | same ($\alpha = 1/(\mu k^2 M)$). | **Adopt**. |
| Viscosity | implicit $\mu = 1$. | explicit `μ`, validated $> 0$. | **Diverge** — downstream users plug in their physical $\mu$ without non-dimensionalising. |
| FFT call sequencing | three r2c, `cufcm_flow_solve`, three c2r. | identical via `mul!`. | **Adopt**. |
| Error handling | each `cufft*Exec*` checks a status code. | none — FFTW.jl `mul!` is infallible once the plan is built. | **Keep** — validate cold, trust hot. |

The differences with arithmetic consequence are the explicit-$\mu$ parameterisation, the
precomputed wavenumber vectors, the single-buffer `fluid_hat`, and the clean
$\boldsymbol{k} = \boldsymbol{0}$ early exit. The rest are independent or CPU-vs-GPU
organisation choices.
