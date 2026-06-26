# Pairwise Correction

Step 6 of the Fast FCM algorithm (Su & Keaveny 2024, §4).

## Summary

Standard FCM is equivalent to the application of a mobility operator $\mathcal{M}^{\mathcal{VF}}$. The
operator resulting from the modified Gaussian kernel is denoted $\tilde{\mathcal{M}}^{\mathcal{VF}}$, from
which the error incurred by widening the Gaussian kernel is summarised by the splitting
$$
    \mathcal{M}^{\mathcal{VF}}
    = \tilde{\mathcal{M}}^{\mathcal{VF}}
    + \left(\mathcal{M}^{\mathcal{VF}} - \tilde{\mathcal{M}}^{\mathcal{VF}}\right) \text{.}
$$
Thus, the difference operator $\left(\mathcal{M}^{\mathcal{VF}} - \tilde{\mathcal{M}}^{\mathcal{VF}}\right)$ is the correction term that needs to be applied to recover the approximation properties
of FCM. This operator is approximately sparse due to the decaying nature of the Gaussian kernel,
and therefore we apply it as a sparse pair-wise sum using the particle binning in the cells.

The analytical form of the correction is given, for the block corresponding to particles $n$ and $m$,
$$\begin{align*}
    \boldsymbol{M}_{nm} - \tilde{\boldsymbol{M}}_{nm} &= \left(\boldsymbol{G}\left( \boldsymbol{Y}_n - \boldsymbol{Y}_m \right) + \sigma^2 \Delta \boldsymbol{G}\left( \boldsymbol{Y}_n - \boldsymbol{Y}_m \right)\right) \left( \operatorname{erf}\left(\frac{\left\lvert \boldsymbol{Y}_n - \boldsymbol{Y}_m \right\rvert}{\sigma \sqrt{2}}\right) - \operatorname{erf}\left(\frac{\left\lvert \boldsymbol{Y}_n - \boldsymbol{Y}_m \right\rvert}{\Sigma \sqrt{2}}\right) \right) \\
    &\qquad + \boldsymbol{S}^{(3)}\left( \boldsymbol{Y}_n - \boldsymbol{Y}_m; \sigma \sqrt{2} \right) - \boldsymbol{S}^{(3)}\left( \boldsymbol{Y}_n - \boldsymbol{Y}_m; \Sigma \sqrt{2} \right) \\
    &\qquad - \left(\sigma^2 - \Sigma^2\right) \boldsymbol{Q}^{(2)}\left( \boldsymbol{Y}_n - \boldsymbol{Y}_m; \Sigma \sqrt{2} \right) - \frac{\left(\sigma^2 - \Sigma^2\right)^2}{4} \boldsymbol{T}\left( \boldsymbol{Y}_n - \boldsymbol{Y}_m; \Sigma \sqrt{2} \right) \quad \text{(§3, equation (31))} \text{,}
\end{align*}$$
where $\boldsymbol{G}$ is the Stokeslet (Oseen tensor), $\Delta\boldsymbol{G}$ its Laplacian, and $\boldsymbol{S}^{(3)}$, $\boldsymbol{Q}^{(2)}$, $\boldsymbol{T}$ are operators of the FCM mobility expansion; all are defined in Method.

The self correction is well defined analytically,
$$
    \lim_{\left\lvert\boldsymbol{Y}_n - \boldsymbol{Y}_m\right\rvert \to 0} \boldsymbol{M}_{nm} - \tilde{\boldsymbol{M}}_{nm} = \underbrace{\left( \frac{1}{6\pi\mu a} - \frac{1}{6\pi\mu\Sigma\sqrt{\pi}} + \frac{\sigma^2 - \Sigma^2}{12\mu \left(\Sigma \sqrt{\pi}\right)^3} - \frac{\left(\sigma^2 - \Sigma^2\right)^2}{32\mu \Sigma^5 \pi^{\frac{3}{2}}} \right)}_{=:\ \delta} \boldsymbol{I} \text{.}
$$

The code defines
- `self_correction_term` $= \delta$,
- `isotropic_coefficient` $= A$ and `parallel_coefficient` $= B$ — the two pair scalars
  of the correction tensor $A\boldsymbol{I} + B\boldsymbol{x} \otimes \boldsymbol{x}$ (Method).

## Method

### The correction as a difference of two FCM mobilities

The FCM pairwise mobility relating the force on particle $m$ to the velocity of particle
$n$ is the regularised Stokeslet at the width $\sigma\sqrt2$,
$$
\boldsymbol{M}^{\mathcal{VF}}_{nm} = \boldsymbol{S}(\boldsymbol{Y}_n - \boldsymbol{Y}_m;\ \sigma\sqrt2)
\qquad (\text{§2, equation (14)}),
$$
and the modified-kernel mobility expands as
$$
\tilde{\boldsymbol{M}}^{\mathcal{VF}}_{nm}
= \boldsymbol{S}(\Sigma\sqrt2) + (\sigma^2 - \Sigma^2)\boldsymbol{Q}(\Sigma\sqrt2)
+ \frac{(\sigma^2 - \Sigma^2)^2}{4}\boldsymbol{T}(\Sigma\sqrt2),
$$
with $\boldsymbol{S} = \boldsymbol{S}^{(1)} + \boldsymbol{S}^{(2)} + \boldsymbol{S}^{(3)}$ and
$\boldsymbol{Q} = \boldsymbol{Q}^{(1)} + \boldsymbol{Q}^{(2)}$ (the operators
$\boldsymbol{S}^{(1\text{--}3)}$ are §2, equations (8)–(10); $\boldsymbol{Q}^{(1,2)}$ are
equations (16)–(17); $\boldsymbol{T}$ is §3, equation (30); the expansion itself is
equation (29)). Their difference is the
correction of the Summary. The package evaluates the paper's regrouped closed form directly:
the $(\boldsymbol{G} + \sigma^2\Delta\boldsymbol{G})$ grouping analytically pre-cancels the
`erf` of $\boldsymbol{Q}^{(1)}$, so only $\boldsymbol{Q}^{(2)}$ and $\boldsymbol{T}$ (both
Gaussian-only) survive as explicit terms. The transcendental cost is the same either way —
two `erf` and two `exp` per interaction.

### Constituent operators (the closed forms the code uses)

Each building block has the form $c_I\boldsymbol{I} + c_{xx}\boldsymbol{x} \otimes \boldsymbol{x}$
with an **un-normalised** $\boldsymbol{x} \otimes \boldsymbol{x}$ (this avoids a division by $r^2$).
Write $\boldsymbol{x} = \boldsymbol{Y}_n - \boldsymbol{Y}_m$, $r = \lVert\boldsymbol{x}\rVert$, and
the $\sqrt2$-scaled Gaussian $\Delta_w \equiv \Delta(\boldsymbol{x}; w\sqrt2) = (4\pi w^2)^{-3/2}e^{-r^2/4w^2}$.
The Stokeslet and its Laplacian are
$$
\boldsymbol{G}(\boldsymbol{x}) = \frac{1}{8\pi\mu}\left(\frac{\boldsymbol{I}}{r} + \frac{\boldsymbol{x} \otimes \boldsymbol{x}}{r^3}\right),
\qquad
\Delta\boldsymbol{G}(\boldsymbol{x}) = \frac{1}{4\pi\mu}\left(\frac{\boldsymbol{I}}{r^3} - \frac{3\boldsymbol{x} \otimes \boldsymbol{x}}{r^5}\right)
\qquad (\boldsymbol{G} \text{ is equation (11)};\ \Delta\boldsymbol{G} \text{ follows from equations (9), (13)}).
$$

The closed forms in the table below — the coefficients of $\boldsymbol{S}^{(3)}$,
$\boldsymbol{Q}^{(2)}$, $\boldsymbol{T}$ and the Laplacian $\Delta\boldsymbol{G}$ — are
verified term-by-term against the published equations (§2, equations (10), (17) and §3,
equation (30); $\eta = \mu$; each block is evaluated at its $\sqrt2$-scaled width, so
$\Delta(\boldsymbol{x}; w\sqrt2) = \Delta_w$). They also reproduce the difference of full
mobilities independently, which the code's `_correction_scalars` is checked against
(verification test 2).

| Block | $c_I$ | $c_{xx}$ |
|---|---|---|
| $\boldsymbol{G}$ | $\dfrac{1}{8\pi\mu r}$ | $\dfrac{1}{8\pi\mu r^3}$ |
| $\sigma^2\Delta\boldsymbol{G}$ | $\dfrac{\sigma^2}{4\pi\mu r^3}$ | $-\dfrac{3\sigma^2}{4\pi\mu r^5}$ |
| $\boldsymbol{S}^{(3)}(w\sqrt2)$ | $-\dfrac{2w^4}{\mu r^2}\Delta_w$ | $+\dfrac{6w^4}{\mu r^4}\Delta_w$ |
| $\boldsymbol{Q}^{(2)}(\Sigma\sqrt2)$ | $-\dfrac{1}{\mu}\left(1+\dfrac{2\Sigma^2}{r^2}\right)\Delta_\Sigma$ | $+\dfrac{1}{\mu r^2}\left(1+\dfrac{6\Sigma^2}{r^2}\right)\Delta_\Sigma$ |
| $\boldsymbol{T}(\Sigma\sqrt2)$ | $\dfrac{1}{2\mu\Sigma^2}\left(2-\dfrac{r^2}{2\Sigma^2}\right)\Delta_\Sigma$ | $\dfrac{1}{4\mu\Sigma^4}\Delta_\Sigma$ |

### Collapse to two pair scalars

Every block is isotropic in the same two tensors, $\boldsymbol{I}$ and
$\boldsymbol{x} \otimes \boldsymbol{x}$, so the whole correction collapses to
$$
\boldsymbol{M}_{nm} - \tilde{\boldsymbol{M}}_{nm} = A(r)\boldsymbol{I} + B(r)\boldsymbol{x} \otimes \boldsymbol{x},
$$
where $A$ collects every $c_I$ and $B$ every $c_{xx}$ from the difference of mobilities. The
$(\boldsymbol{G} + \sigma^2\Delta\boldsymbol{G})$ group carries the `erf` factor; the two
$\boldsymbol{S}^{(3)}$ terms contribute $\Delta_\sigma$ and $\Delta_\Sigma$ pieces; and the
$-(\sigma^2 - \Sigma^2)\boldsymbol{Q}^{(2)}$ and $-\tfrac{(\sigma^2-\Sigma^2)^2}{4}\boldsymbol{T}$
terms contribute further $\Delta_\Sigma$ pieces. Writing
$\mathrm{erf}_{2w} = \operatorname{erf}(r/2w)$, this gives
$$
A(r) = (\mathrm{erf}_{2\sigma}-\mathrm{erf}_{2\Sigma})\left(\frac{1}{8\pi\mu r}+\frac{\sigma^2}{4\pi\mu r^3}\right)
- \frac{2\sigma^4}{\mu r^2}\Delta_\sigma
+ \left[\frac{2\Sigma^4}{\mu r^2}+\frac{(\sigma^2-\Sigma^2)}{\mu}\left(1+\frac{2\Sigma^2}{r^2}\right)-\frac{(\sigma^2-\Sigma^2)^2}{4}\cdot\frac{2-r^2/2\Sigma^2}{2\mu\Sigma^2}\right]\Delta_\Sigma ,
$$
$$
B(r) = (\mathrm{erf}_{2\sigma}-\mathrm{erf}_{2\Sigma})\left(\frac{1}{8\pi\mu r^3}-\frac{3\sigma^2}{4\pi\mu r^5}\right)
+ \frac{6\sigma^4}{\mu r^4}\Delta_\sigma
+ \left[-\frac{6\Sigma^4}{\mu r^4}-\frac{(\sigma^2-\Sigma^2)}{\mu r^2}\left(1+\frac{6\Sigma^2}{r^2}\right)-\frac{(\sigma^2-\Sigma^2)^2}{4}\cdot\frac{1}{4\mu\Sigma^4}\right]\Delta_\Sigma .
$$
The velocity correction to particle $n$ from the force $\boldsymbol{F}_m$ on a neighbour $m$
is then
$$
\Delta\boldsymbol{V}_n \mathrel{+}= A(r)\boldsymbol{F}_m + B(r)(\boldsymbol{x}\cdot\boldsymbol{F}_m)\boldsymbol{x}.
$$
Collapsing to the two scalars $A, B$ replaces a dense $3\times3$ matrix–vector product with
two scalar evaluations, and the $\Sigma\sqrt2$ `erf` and Gaussian are shared across the
$\boldsymbol{Q}^{(2)}$ and $\boldsymbol{T}$ terms, so only two `erf` and two `exp` are
evaluated per interaction.

### The argument of the error function

The printed correction (Summary, equation (31)) shows the factor
$\operatorname{erf}(r/(\sigma\sqrt2)) - \operatorname{erf}(r/(\Sigma\sqrt2))$, but the
correct argument is $\operatorname{erf}(r/(2\sigma)) - \operatorname{erf}(r/(2\Sigma))$:
the printed equation (31) carries a typo in the `erf` argument. To see it, the regularised
Stokeslet is $\boldsymbol{S}^{(1)}(\boldsymbol{x}; \sigma) = \operatorname{erf}(r/(\sigma\sqrt2))\boldsymbol{G}$
(equation (12)), and the pairwise mobility uses the $\sqrt2$-scaled width,
$\boldsymbol{M}^{\mathcal{VF}}_{nm} = \boldsymbol{S}(\cdot; \sigma\sqrt2)$ (equation (14)),
so $\boldsymbol{S}^{(1)}$ inside $\boldsymbol{M}^{\mathcal{VF}}$ carries
$\operatorname{erf}(r/(2\sigma))$. Collecting the $\Delta\boldsymbol{G}$ terms from
$\boldsymbol{S}^{(2)}(\sigma\sqrt2) - \boldsymbol{S}^{(2)}(\Sigma\sqrt2) - (\sigma^2-\Sigma^2)\boldsymbol{Q}^{(1)}(\Sigma\sqrt2)$
then yields the coefficient $\sigma^2(\mathrm{erf}_{2\sigma} - \mathrm{erf}_{2\Sigma})$, fixing
the argument as $r/(2w)$. The package implements $r/(2w)$, consistent with the regularised
Stokeslet and the author's reference implementation; the independent difference-of-mobilities
oracle (verification test 2) would fail otherwise.

### The self term ($r \to 0$)

At $r = 0$ the off-diagonal $\boldsymbol{x} \otimes \boldsymbol{x}$ part vanishes and the
correction is the well-defined diagonal $\delta\boldsymbol{I}$ of the Summary, with
$a = \sigma\sqrt\pi$,
$$
\delta = \frac{1}{6\pi\mu a} - \frac{1}{6\pi\mu(\Sigma\sqrt\pi)}
+ \frac{\sigma^2-\Sigma^2}{12\mu(\Sigma\sqrt\pi)^3}
- \frac{(\sigma^2-\Sigma^2)^2}{32\mu\Sigma^5\pi^{3/2}}.
$$
It is added to every particle independent of its neighbours.

### Symmetry and the $\Sigma = \sigma$ limit

Because $A$ and $B$ depend only on $r$ and the tensor $A\boldsymbol{I} + B\boldsymbol{x} \otimes \boldsymbol{x}$
is symmetric, the pair correction is self-adjoint ($\boldsymbol{M}_{nm} = \boldsymbol{M}_{mn}^T$),
so the splitting preserves the symmetric positive-definite structure of the mobility (see
[mobility.md](mobility.md)). At $\Sigma = \sigma$ every Gaussian term and the `erf` difference
vanish, and $\delta = 0$, so the correction is identically zero — consistent with
$\tilde{\mathcal{M}}^{\mathcal{VF}} = \mathcal{M}^{\mathcal{VF}}$ there.

## Contract

### Cold-path input and derived state

None new. The correction reuses the cell geometry (`R_c`, `num_cells`, `cell_size`) and the
kernel widths (`σ`, `Σ`, `a`, `μ`) already on `FFCMConfig`; $\delta$ is recomputed once per
call (a few flops, negligible beside the pair loop), keeping the struct unchanged. The
constructor enforces one precondition this step relies on: $R_c \leq \min(L)/2$, so the
minimum-image separation is unambiguous and no particle is corrected against its own
periodic image.

### Hot-path input (per `correct_velocities!` call)

- `config.particles.Y_sorted`, `config.particles.F_sorted` — sorted positions and forces (step 2).
- `config.cells.cell_start`, `config.cells.cell_end` — per-cell sorted-slot ranges (step 2).
- `config.cells.original_index` — sorted slot to original particle index.
- Positions folded into $[0, L_i)$ by `wrap_positions!` (step 1).

### Hot-path output

- `V` (`AbstractMatrix{T}`, shape `(3, N)`, caller-owned, original order) — the same buffer
  `interpolate_velocities!` wrote. `correct_velocities!` **adds** the correction in place
  (`V[:, n] += ΔV_n`), realising
  $\mathcal{M} = \tilde{\mathcal{M}} + (\mathcal{M} - \tilde{\mathcal{M}})$.

### Side effects

None beyond writing `V`. `Y_sorted`, `F_sorted`, and the cell list are read-only; the
per-particle accumulation lives in registers / `SVector`s.

### Periodicity contract

Each pair interacts through its nearest periodic image only: the separation
$\boldsymbol{Y}_n - \boldsymbol{Y}_m$ is reduced to $[-L_i/2, L_i/2]$ per axis by `_min_image`,
and the 27-cell neighbour sweep wraps cell coordinates $\bmod\ m_i$. The $R_c \leq \min(L)/2$
precondition guarantees a single relevant image and excludes self-image corrections.

### Boundary cases

- **No neighbours within $R_c$** — only the self term is added ($\boldsymbol{V}_n \mathrel{+}= \delta\boldsymbol{F}_n$).
- **$\Sigma = \sigma$** — $A = B = 0$ and $\delta = 0$; `V` is unchanged.
- **Coincident distinct particles ($r \to 0$)** — assumed not to occur; the pair scalars are
  evaluated only for $0 < r < R_c$, and the $r = 0$ case is the separate self term.

## Implementation

`correct_velocities!(V, config)` is the hot-path entry. It reads `config.particles.Y_sorted`,
`config.particles.F_sorted`, the cell list, and `config.cells.original_index`, and **adds** the pairwise and
self corrections in place to `V`, returning `V`. It is allocation-free and type-stable.

It delegates to the function-barrier kernel `_correct_velocities_kernel!`, which takes naked
arrays and scalars. The kernel computes $\delta$ once, then walks the cells in increasing
linear hash. For each sorted particle it seeds the self term $\delta\boldsymbol{F}$, sweeps
the 27 wrapped neighbour cells, and for every neighbour within $R_c$ (minimum image)
accumulates $A(r)\boldsymbol{F}_m + B(r)(\boldsymbol{x}\cdot\boldsymbol{F}_m)\boldsymbol{x}$,
scattering the result to `V[:, original_index[s]]` via `+=`. Each particle writes only its own
output column — a gather, with no write race.

Three pure helpers carry the physics: `_correction_scalars(r, σ, Σ, μ)` returns the pair
scalars $(A, B)$; `_self_correction(σ, Σ, a, μ)` returns $\delta$; and `_min_image(x, L)`
reduces a separation `SVector{3}` to its nearest image by
$x_i - L_i\operatorname{round}(x_i/L_i)$. Round-to-nearest-ties-to-even makes the reduction
exactly antisymmetric, $\texttt{\_min\_image}(-x) = -\texttt{\_min\_image}(x)$, which keeps the
pair correction symmetric.

| Phase | Allocations | Functions |
|---|---|---|
| Cold | — | `R_c ≤ min(L)/2` validation in the constructor; no new fields. |
| Hot  | `@ballocated == 0` | `correct_velocities!(V, config)` (after `interpolate_velocities!`). |

The hot path is allocation-free and type-stable on `T <: AbstractFloat`.

## Performance notes

- **Gather, not half-list — a flagged tradeoff.** Each particle accumulates its own
  correction by reading every neighbour within $R_c$ over the full 27 cells, writing only its
  own column. This evaluates each ordered pair's scalars twice ($\approx 2\times$ the
  `erf`/`exp` of a half-neighbour list), but it has **no write race**: threads or SIMD
  parallelise trivially per particle, exactly like the interpolation gather. The half-list
  with a Newton's-third-law dual update would halve the transcendental count at the cost of a
  write race; revisit it only if a benchmark shows the pair loop dominates.
- **Two-scalar collapse.** $A\boldsymbol{I} + B\boldsymbol{x} \otimes \boldsymbol{x}$ instead of a
  dense $3\times3$ matrix–vector product; the $\Sigma\sqrt2$ `erf` and Gaussian are shared
  across the $\boldsymbol{Q}^{(2)}$ and $\boldsymbol{T}$ terms, so only two `erf` and two `exp`
  run per interaction.
- **Self term once per call**, at kernel entry — not stored on `config`, not recomputed per
  particle.
- **Contiguous cell walk.** The outer loop iterates cells in increasing linear hash, so
  `Y_sorted`/`F_sorted` columns are read in their sorted (contiguous-per-cell) order; the
  per-particle output is scattered to `V[:, original_index[s]]`.
- **`@inbounds` after a once-per-call bounds argument.** The inner cutoff branch
  ($r^2 < R_c^2$) blocks a naive `@simd`, so vectorisation is deferred to a benchmark.

## Verification

Relative comparisons to a non-zero reference use `rtol = sqrt(eps(T))`; quantities that
should be $\approx 0$ use `atol = 1e-10` (`Float64`) / `1e-6` (`Float32`); the
truncation-dominated end-to-end test uses the documented $(M_G, \Sigma/h)$ / finite-box
tolerance stated inline.

`test/test_correct_velocities.jl`:

1. **Self term matches the $r \to 0$ limit.** `_self_correction(σ, Σ, a, μ)` equals the
   hand-evaluated $\delta$ over a sweep of $\Sigma/\sigma$ and $\mu$.
2. **Pair scalars reproduce the difference of full mobilities.** For sampled $\boldsymbol{x}$,
   assemble $\boldsymbol{S}(\sigma\sqrt2) - \boldsymbol{S}(\Sigma\sqrt2) - (\sigma^2-\Sigma^2)\boldsymbol{Q}(\Sigma\sqrt2) - \tfrac{(\sigma^2-\Sigma^2)^2}{4}\boldsymbol{T}(\Sigma\sqrt2)$
   as a $3\times3$ matrix from the full operators and assert it equals
   $A(r)\boldsymbol{I} + B(r)\boldsymbol{x} \otimes \boldsymbol{x}$. An algebraically distinct
   oracle that pins the $A, B$ collapse and the `erf` argument.
3. **$\Sigma = \sigma$.** `_self_correction` and `_correction_scalars` both return 0;
   `correct_velocities!` leaves `V` unchanged.
4. **Isolated particle gets only the self term.** One particle (or particles spaced beyond
   $R_c$): $\boldsymbol{V}_n \mathrel{+}= \delta\boldsymbol{F}_n$ exactly.
5. **Two-particle pair correction matches the tensor.** Two particles at $r < R_c$: the added
   velocity equals the difference-of-mobilities tensor (test 2) times the force, for both
   partners.
6. **Pair correction is symmetric.** $\boldsymbol{F}_a^T(\text{corr}\boldsymbol{F}_b) = \boldsymbol{F}_b^T(\text{corr}\boldsymbol{F}_a)$,
   pinning $\boldsymbol{M}_{nm} = \boldsymbol{M}_{mn}^T$ — the symmetry that underpins the
   SPD structure.
7. **Linearity in `F`, translation, and periodicity invariance.** Shifting all positions, or
   one particle by $L\hat{e}_i$, leaves the correction unchanged.
8. **Additive onto `V`.** `correct_velocities!` adds to a pre-filled `V` rather than
   overwriting it.

`test/test_single_sphere_mobility.jl` (extended):

9. **$\Sigma$-independent single-sphere self-mobility (end-to-end, all six steps).** Run the
   full pipeline for one particle at $\Sigma/\sigma \in \{1.5, 3\}$. The corrected
   self-velocity matches the $\sigma$-regularised reciprocal-lattice Stokeslet sum (the same
   reference as the step-5 test), and the two $\Sigma$ runs agree to the documented
   grid-truncation tolerance. This is the proof that the correction removes the
   $\Sigma$-dependence introduced by the grid step.

`test/test_correct_velocities_inferred.jl` — `@inferred` for `correct_velocities!`, the
kernel, `_correction_scalars`, and `_self_correction`. `test/test_correct_velocities_allocations.jl`
— `@ballocated == 0`. `test/test_jet.jl` — `JET.@test_call correct_velocities!`.
`test/test_aqua.jl` — package hygiene (including the `SpecialFunctions` dependency).

## Differences from cuFCM

> Comparison against the C++/CUDA reference implementation, kept for validation during
> development and removed once the port is complete.

Audited `cuFCM/src/CUFCM_CORRECTION.cu` (`cufcm_pair_correction`, `cufcm_self_correction`, and
the per-block coefficient helpers) and `cuFCM/src/CUFCM_SOLVER.cu` (the self-term constants and
cell sizing). Only the active force-only path is considered.

| Facet | cuFCM | This package | Decision |
|---|---|---|---|
| Correction formula | difference of full mobilities | the paper-simplified equation (31), typo-corrected; cuFCM's form is the test oracle | **Keep** the simplified form ($\boldsymbol{Q}^{(1)}$'s `erf` pre-cancelled, marginally leaner); same two `erf` + two `exp` cost. |
| `erf` argument | $\operatorname{erf}(r/2\sigma)$, $\operatorname{erf}(r/2\Sigma)$ | same | **Adopt** — the correct argument; the printed equation (31) is flagged above. |
| Tensor decomposition | $c_I\boldsymbol{I} + c_{xx}\boldsymbol{x} \otimes \boldsymbol{x}$, un-normalised $\boldsymbol{x} \otimes \boldsymbol{x}$ | same | **Adopt** — one fewer division. |
| Viscosity | $\eta = 1$ (omitted) | $\mu$ carried explicitly | **Keep** — matches `stokes_solve!`/`config`. |
| Self term | $\boldsymbol{V} \mathrel{+}= \boldsymbol{F}\cdot(\text{Stokes} - \text{Mod} + \text{PD} - \text{BiLap})$ | `_self_correction` closed form | **Adopt** — equal term-for-term to $\delta$. |
| Neighbour search | 13-neighbour half list + Newton's-third-law dual update with atomic adds | full 27-cell gather, self-write only, no atomics | **Diverge** — no write race; GPU-gather-aligned (see Performance notes). |
| Min-image | $x - L\cdot\operatorname{int}(x/(0.5L))$ (truncation) | $x - L\operatorname{round}(x/L)$ | **Keep** — equivalent for $r < L/2$. |
| Cutoff test | $r^2 < R_c^2$ | same | **Keep**. |
| Cell count | `max(L/R_c, 3)` | `max(floor(L/R_c), 3)` | **Keep** — already matches. |
| Grid-based correction variant | `cufcm_flowfield_correction` spreads a correction force onto the grid | — | **Out of scope** — not the paper's pairwise method. |
| Rotation/torque | `#if ROTATION` blocks | — | **Out of scope** — force → velocity only. |

For the future GPU backend, the paper assigns one thread per particle to the correction and
gathers; the 27-cell gather here maps directly onto that, unlike cuFCM's half-list with atomic
adds.
