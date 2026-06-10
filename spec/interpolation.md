# Interpolation

Step 5 of the Fast FCM algorithm (Su & Keaveny 2024, §4).

## Summary

The fluid velocity $\boldsymbol{u}$ is set to match a no-slip condition on the particles. This
is realised by setting the particle velocities to be interpolations of the fluid velocity.
This operator,
$$\begin{align*}
    \tilde{\mathcal{J}} : \boldsymbol{L}^{2}\left(\Omega\right) &\to \left[\mathbb{R}^3\right]^N \\
    \boldsymbol{u} &\mapsto \int_{\Omega} \boldsymbol{u}\left(\boldsymbol{x}\right) \tilde{\Delta}_n\left(\boldsymbol{x}; \Sigma\right) \, \mathrm{d}\boldsymbol{x} \quad \text{(§3, equation (26))} \text{,}
\end{align*}$$
is the adjoint of the spreading operator $\tilde{\mathcal{J}}^\dagger$ because the spreading and interpolation
kernels are the same modified Gaussian kernel.

The resulting vector $\tilde{\mathcal{J}}\left[\boldsymbol{u}\right] = \left(\boldsymbol{V}_n\right)_{n = 1}^N$ corresponds to the particle velocities and closes the definition of the mobility operator.

Since we have a sampling of $\boldsymbol{u}$ on the gridpoints, the integral is approximated
by the trapezoidal rule on those points using the $M_G \times M_G \times M_G$ stencil.

In the code, we define
- `V` $= \left(\boldsymbol{V}_1, \dots, \boldsymbol{V}_N\right)$.

## Method

### Interpolation as a quadrature

The velocity of particle $n$ is the modified-kernel volume average
$\boldsymbol{V}_n = \int_\Omega \boldsymbol{u}(\boldsymbol{x})\tilde{\Delta}_n(\boldsymbol{x}; \Sigma)\, \mathrm{d}\boldsymbol{x}$.
With $\boldsymbol{u}$ known only at the grid points, the integral is approximated by the
trapezoidal rule with the uniform weight $h^3$ over the same $M_G^3$ stencil as the spread,
$$
\boldsymbol{V}_n = h^3 \sum_{\text{stencil}} \boldsymbol{u}(\boldsymbol{x}_g)
\bigl(a_0 + a_2 r_n^2\bigr) g_x g_y g_z ,
$$
where the modified kernel $\tilde{\Delta}_n = (a_0 + a_2 r_n^2)\Delta_n$, its separable
1-D weights $g_x g_y g_z$, the axis-squared distances $r_n^2$, and the nearest-anchored
stencil are exactly as derived in [force-spreading.md](force-spreading.md). The only
differences from the spread are the direction — a gather from the grid rather than a
scatter to it — and the quadrature factor $h^3$, applied once per particle.

### Why the trapezoidal rule, with the uniform weight $h^3$

Two independent reasons fix both the rule and its weight; neither is a free quadrature
choice.

**1. Exact discrete adjointness makes the mobility symmetric.**
Let $S$ be the spread matrix, with entry $S_{g,n} = \tilde{\Delta}_n(\boldsymbol{x}_g; \Sigma)$
the weight particle $n$ contributes to grid point $g$ — so `spread_forces!` computes the
grid field $\sum_n S_{g,n}\boldsymbol{F}_n$. The trapezoidal interpolation above is
$\boldsymbol{V}_n = h^3 \sum_g S_{g,n}\boldsymbol{u}(\boldsymbol{x}_g)$, i.e. the matrix
$h^3 S^T$ acting on the grid velocity. Writing $P$ for the step-2 gather permutation
($\boldsymbol{F}_{\text{sorted}} = P\boldsymbol{F}$, undone on output by the scatter-back
$P^T$) and $\mathcal{L}^{-1}$ for the Stokes solve, the assembled smooth mobility is
$$
\tilde{\mathcal{M}}^{\mathcal{V}\mathcal{F}} = h^3 (SP)^T \mathcal{L}^{-1} (SP).
$$
$\mathcal{L}^{-1}$ is symmetric: per Fourier mode it is the orthogonal projector
$\boldsymbol{I} - \hat{\boldsymbol{k}}\hat{\boldsymbol{k}}^T$ (symmetric, idempotent,
eigenvalues $0, 1, 1$) scaled by the positive factor $1/(\mu k^2)$, so it is symmetric and
positive-semidefinite. Therefore $\tilde{\mathcal{M}}^{\mathcal{V}\mathcal{F}}$ is symmetric
and positive-semidefinite for **any** $S$, and positive-definite on the force space for
physical (distinct-particle) configurations. This holds **only** because interpolation is
the exact transpose of spreading: the uniform weight $h^3$ at every stencil point is what
makes $\tilde{\mathcal{J}} = h^3 S^T$ exactly. A non-uniform weight would replace $h^3$
by a diagonal $W \neq h^3 \boldsymbol{I}$, and Gauss-type nodes would change the node set
entirely; either breaks $\tilde{\mathcal{J}} = h^3 S^T$, the transpose relation, and the
symmetry of the mobility. The downstream iterative resistance solver depends on that
symmetry (conjugate gradients assumes a symmetric positive-definite operator), so the
uniform-weight trapezoidal rule is a structural requirement, not an accuracy preference.

**2. Spectral accuracy on the periodic grid (Euler–Maclaurin).**
On a periodic domain the trapezoidal rule is far more accurate than its nominal
second order: for a smooth integrand it converges faster than any power of $h$. The
mechanism is the Euler–Maclaurin formula, which for a function $\phi$ on $[0, L]$ sampled
at the grid points $x_j = j h$ relates the trapezoidal sum to the exact integral by
$$
h \sum_{j} \phi(x_j) - \int_0^L \phi(x) \, \mathrm{d}x
= \sum_{m \geq 1} \frac{B_{2m}}{(2m)!} h^{2m}
\left[\phi^{(2m-1)}(L) - \phi^{(2m-1)}(0)\right],
$$
with $B_{2m}$ the Bernoulli numbers. The error is a sum of **boundary** terms involving odd
derivatives at the two endpoints. For an $L$-periodic integrand every derivative matches at
the endpoints, $\phi^{(k)}(L) = \phi^{(k)}(0)$, so every term in the series cancels and the
error decays faster than any fixed power of $h$ — "spectral", or super-algebraic,
convergence. The integrand here is $\boldsymbol{u}(\boldsymbol{x})\tilde{\Delta}_n(\boldsymbol{x}; \Sigma)$,
a product of two $C^\infty$, $L$-periodic functions (the discrete velocity field is a
trigonometric polynomial; the Gaussian kernel, summed over its periodic images, is smooth
and $L$-periodic), so the full-grid trapezoidal sum integrates it to spectral accuracy.
Restricting the sum to the $M_G^3$ stencil drops only the Gaussian tail outside the
stencil, which is exponentially small in $M_G$. Both error sources are therefore
negligible at modest $M_G$.

## Contract

### Cold-path input and derived state

None new. Interpolation reuses the grid, kernel widths, and per-axis scratch vectors built
for steps 3 and 4. It runs strictly after the spread and the Stokes solve, so reusing the
scratch is safe.

### Hot-path input (per `interpolate_velocities!` call)

- `config.fluid_velocity` — the fluid velocity field $\boldsymbol{u}(\boldsymbol{x}_g)$ from
  `stokes_solve!`.
- `config.Y_sorted` — sorted particle positions (step 2), folded into $[0, L_i)$.
- `config.original_index` — sorted slot to original particle index (step 2), for the
  scatter-back to the caller's order.

### Hot-path output

- `V` (`AbstractMatrix{T}`, shape `(3, N)`), **caller-owned, original particle order**. On
  return `V[:, n]` holds $\boldsymbol{V}_n$ in the caller's indexing. This terminal grid
  step produces the user-facing result, so it writes the output buffer directly rather than
  a `config` field, and `(3, N)` is the shape the matrix-free `mul!` pipeline needs.

### Side effects

- The per-axis scratch vectors are overwritten with the last particle's values; they carry
  no between-call invariant (shared with step 3).
- `config.fluid_velocity` is **not** modified.

### Periodicity contract

Identical to the spread: positions enter folded into $[0, L_i)$; each particle's stencil is
wrapped $\bmod\ M_i$; a particle near the edge gathers from a stencil that wraps to the
opposite side. The cold-path precondition $M_G \leq \min(M_x, M_y, M_z)$ (validated in the
constructor; see [force-spreading.md](force-spreading.md)) guarantees the wrapped stencil
indices are distinct on each axis, so the quadrature gathers each grid point at most once —
a stencil wider than the grid would double-weight a grid point in this adjoint of the spread.

### Boundary cases

- **Particle exactly on a grid point** — distance 0 along each axis; no special case.
- **$\Sigma = \sigma$** — $a_0 = 1$, $a_2 = 0$, so $\tilde{\Delta}_n = \Delta_n$ and
  interpolation is the plain-Gaussian FCM volume average.

## Implementation

`interpolate_velocities!(V, config)` is the hot-path entry. It reads `config.fluid_velocity`,
`config.Y_sorted`, and `config.original_index`, and writes the particle velocities into `V`
in the caller's original order, returning `V`. It is allocation-free and type-stable.

It delegates to the function-barrier kernel `_interpolate_velocities_kernel!`, which takes
naked arrays and scalars. Per sorted particle the kernel fills the stencil scratch (via the
shared `_fill_particle_stencil!`, identical to the spread), accumulates the gather
$\sum_{\text{stencil}} \boldsymbol{u}(\boldsymbol{x}_g)(a_0 + a_2 r_n^2)g_x g_y g_z$ into
scalar accumulators, scales by $h^3$ once, and writes the result to `V[:, original_index[s]]`.
That scatter-back through `original_index` is the inverse of the step-2 gather — the
$P^T$ of Method — so the output lands in the caller's order with no extra buffer or pass.

Because `_fill_particle_stencil!` is shared verbatim with `_spread_forces_kernel!`, the
interpolation weights are bit-for-bit the spread weights, which is what makes
$\tilde{\mathcal{J}} = h^3 S^T$ hold exactly. The stencil index buffers are `idx_x/y/z`.

| Phase | Allocations | Functions |
|---|---|---|
| Cold | — | No new cold-path work; reuses steps 3 and 4 state. |
| Hot  | `@ballocated == 0` | `interpolate_velocities!(V, config)` (after `spread_forces!` → `stokes_solve!`). |

The hot path is allocation-free and type-stable on `T <: AbstractFloat`.

## Performance notes

- **Stride-1 reads.** Destructuring `(ux, uy, uz) = components(fluid_velocity)` lets the
  inner stencil loop read `ux[i_x, i_y, i_z]` stride-1 along $i_x$, mirroring the spread
  write loop.
- **Scalar accumulation.** The stencil sum accumulates into scalars `vx, vy, vz`; the
  `SVector{3, T}` is written to `V[:, n]` once per particle, scaled by $h^3$.
- **$h^3$ once per particle**, not folded per grid point — three multiplies per particle
  versus three per stencil point. Algebraically identical, and it keeps the discrete-adjoint
  test bit-exact against the spread weights.
- **Scratch reuse.** The `gaussian_*`, `r²_*`, `idx_*` vectors built for the spread are reused;
  interpolation runs strictly after spread + solve, so there is no overlap.
- **No write race.** The gather only reads the grid, and the per-particle output columns are
  distinct (a permutation), so this loop is a clean future threading / SIMD target — deferred
  to a benchmark.

## Verification

Relative comparisons to a non-zero reference use `rtol = sqrt(eps(T))`; quantities that
should be $\approx 0$ use `atol = 1e-10` (`Float64`) / `1e-6` (`Float32`);
truncation-dominated tests (constant-flow, single-sphere drag) use the documented
$(M_G, \Sigma/h)$ / finite-box tolerance stated inline.

`test/test_interpolate_velocities.jl`:

1. **Output shape and scatter-back order.** `V` is `(3, N)`; with particles supplied in
   scrambled order, `V` comes out in the original (caller) order — checked against the
   `original_index` permutation.
2. **Discrete adjoint of spreading.** $\langle \tilde{\mathcal{J}}(\boldsymbol{u}), \boldsymbol{e}_n\rangle = h^3\langle \boldsymbol{u}, \tilde{\mathcal{J}}^\dagger(\boldsymbol{e}_n)\rangle$
   for a chosen grid field $\boldsymbol{u}$ and each particle $n$ — pins
   $\tilde{\mathcal{J}} = h^3 S^T$.
3. **Mobility symmetry.** $\boldsymbol{F}_a^T \mathcal{M} \boldsymbol{F}_b = \boldsymbol{F}_b^T \mathcal{M} \boldsymbol{F}_a$
   for random $\boldsymbol{F}_a, \boldsymbol{F}_b$ through the full spread → solve →
   interpolate pipeline — pins the symmetry that the SPD precondition rests on.
4. **Single-particle closed form.** One particle at a known position; set `fluid_velocity`
   to a known analytic field; hand-compute $\boldsymbol{V}_n = h^3 \sum_{\text{stencil}} \boldsymbol{u}(\boldsymbol{x}_g)\tilde{\Delta}_n(\boldsymbol{x}_g)$.
5. **Constant flow → constant velocity.** Uniform `fluid_velocity` $= \boldsymbol{U}$ gives
   $\boldsymbol{V}_n \approx \boldsymbol{U}$ for every particle, within the
   $(M_G, \Sigma/h)$ truncation tolerance. This holds because the kernel integrates to one,
   $\int_{\mathbb{R}^3} \tilde{\Delta}_n = 1$ (derived in force-spreading.md), so a constant
   field is reproduced exactly up to the stencil truncation.
6. **Linearity in the flow field.**
7. **Periodicity.** Particle at $\boldsymbol{Y}_n$ versus $\boldsymbol{Y}_n + L\hat{e}_i$ on
   the same `fluid_velocity` gives identical $\boldsymbol{V}_n$.
8. **$\Sigma = \sigma$ limit.** Interpolation uses the plain Gaussian ($a_0 = 1$,
   $a_2 = 0$), matching the closed-form FCM volume average.

`test/test_single_sphere_mobility.jl`:

9. **Single-sphere periodic self-mobility (end-to-end).** One particle, force
   $\boldsymbol{F}$, $\Sigma/\sigma = 1$, full pipeline. Compare $\boldsymbol{V}$ to the
   continuum periodic self-mobility — the reciprocal-lattice sum
   $\tilde{\boldsymbol{V}}_{\text{self}} = \tfrac{1}{L^3}\sum_{\boldsymbol{k}\neq 0} \tfrac{e^{-\sigma^2 k^2}}{\mu k^2}(\boldsymbol{I} - \hat{\boldsymbol{k}}\hat{\boldsymbol{k}}^T)\boldsymbol{F}$
   (the Fourier form of the regularised Stokeslet, §3, equations (32)–(33) and Appendix A, convolved with the kernel on
   both sides), summed over the reciprocal lattice $\boldsymbol{k} = (2\pi/L)\boldsymbol{n}$
   up to where $e^{-\sigma^2 k^2}$ falls below a documented cutoff. Tolerance is the
   grid-discretisation / aliasing error for the chosen $\Sigma/h$, documented inline. First
   test exercising all five steps composed.

`test/test_interpolate_velocities_inferred.jl` — `@inferred` for the wrapper and the
kernel. `test/test_interpolate_velocities_allocations.jl` — `@ballocated == 0`.
`test/test_jet.jl` — `JET.@test_call interpolate_velocities!`. `test/test_aqua.jl` — package
hygiene.

## Differences from cuFCM

> Comparison against the C++/CUDA reference implementation, kept for validation during
> development and removed once the port is complete.

Audited `cuFCM/src/CUFCM_FCM.cu` (`cufcm_particle_velocities_bpp_shared_dynamic`) and
`cuFCM/src/CUFCM_SOLVER.cu` (`FCM_solver::gather`). Only the active force-only path is
considered. cuFCM independently confirms the adjoint constraint: its spread weight is
$g_x g_y g_z (1 + \cdots)$ with **no** $h^3$, and its interpolate weight is the *identical*
term times $h^3$ — so it too implements $\tilde{\mathcal{J}} = h^3\tilde{\mathcal{J}}^{\dagger\top}$
exactly.

| Facet | cuFCM | This package | Decision |
|---|---|---|---|
| Quadrature weight | $h^3$ folded into the per-grid-point factor | $h^3$ applied once per particle to the accumulated velocity | **Keep** (diverge on location) — algebraically identical, cheaper, and keeps the adjoint test bit-exact. |
| Stencil precompute | nearest-anchor, norm, gauss, wrap — byte-identical to its spread kernel | identical; shared with the spread via `_fill_particle_stencil!` | **Adopt** — reinforces the shared-helper extraction. |
| Polynomial factor | `1 + temp3·r² − temp4` | $(a_0 + a_2 r^2)$ | **Keep** — algebraically identical (same as spread). |
| $r^2$ handling | stores signed `xdis/ydis/zdis`, recomputes $r^2$ | stores $r^2$ directly | **Keep** the $r^2$-only precompute. |
| Velocity-grid read | three SoA arrays | `components(fluid_velocity)`, column-major | **Adopt** — same SoA layout, `StructArray` wrapper. |
| Stencil reduction | block reduce over threads | serial scalar accumulation | **Keep** serial for the CPU MVP; block-reduce is the GPU pattern. |
| Output / unsort | writes in sorted order; unsort handled separately | writes `V[:, original_index[s]]` directly (inverse permutation folded in) | **Diverge** — consistent with materialising sorted arrays in step 2; the fold is exactly $P^T$, preserving adjointness. |
| Rotation/dipole | `rotation == 1` branch | none — force-only | **Out of scope.** |
| `USE_REGULARFCM` | compile-time $\Sigma = \sigma$ branch | no flag; the $\Sigma = \sigma$ limit subsumes it | **Subsume via the limit.** |

For the future GPU backend, interpolation is the gather mirror of the spread's scatter (one
block per particle, threads over the $M_G^3$ stencil, a block reduce to sum). Unlike the
spread's atomic adds, the gather has no write race.
