# Force Spreading

Step 3 of the Fast FCM algorithm (Su & Keaveny 2024, §4).

## Summary

The spreading operator $\tilde{\mathcal{J}}^\dagger$ is applied to the forces $\left(\boldsymbol{F}\right)_{n = 1}^N$ to create the right-hand-side for the Stokes problem. This operator is defined as
$$\begin{align*}
    \tilde{\mathcal{J}}^\dagger : \left[\mathbb{R}^3\right]^N &\to \boldsymbol{L}^{2}\left(\Omega\right) \\
    \left(\boldsymbol{F}\right)_{n = 1}^N &\mapsto \sum_{n = 1}^N \boldsymbol{F}_n \tilde{\Delta}_n\left(\cdot; \Sigma\right) \quad \text{(§3, equation (25))} \text{,}
\end{align*}$$
where $\tilde{\Delta}_n\left(\cdot; \Sigma\right)$ is the Fast FCM Gaussian kernel with width $\Sigma$, defined as
$$
    \tilde{\Delta}_n\left(\boldsymbol{x}; \Sigma\right) \coloneqq \left(1 + \frac{\sigma^2 - \Sigma^2}{2} \Delta \right) \Delta_n\left(\boldsymbol{x}; \Sigma\right) \quad \text{(§3, equation (22))}\text{.}
$$
$\Delta_n(\boldsymbol{x}; \sigma)$ is, in turn, the original Gaussian kernel (with width $\sigma$, its second argument). That is,
$$
  \Delta_n(\boldsymbol{x}; \sigma) = \frac{1}{\left(2\pi\sigma^2\right)^{\frac{3}{2}}} e^{\frac{-\left\lvert \boldsymbol{x} - \boldsymbol{Y}_n \right\rvert^2}{2\sigma^2}} \quad \text{(§2, equation (1))}\text{,}
$$
where $\boldsymbol{Y}_n$ is the position of particle $n$.

The idea behind this kernel is to use $\Sigma \geq \sigma$, i.e. spread the forces using a relatively large Gaussian kernel. That reduces the refinement requirements of the Stokes solver for accuracy, at the cost of an added error in hydrodynamic interactions that needs to be corrected. However, due to the exponential decay of Gaussians, this correction can be applied to a fraction of the total particle pairs while maintaining accuracy. $\Sigma = \sigma$ reduces to the standard FCM.

The Gaussian kernel width in the original FCM is $\sigma = \frac{a}{\sqrt{\pi}}$, chosen so that FCM recovers the Stokes drag law for a single particle of radius $a$ (§2). Thus, $\sigma$ is set by the particle radius, and $\Sigma$ is variable to accelerate the method. For numerical reasons, it is reasonable to assume the particle radius $a$ as our length scale, so that all dimensions are expressed in terms of $a$.

Distinguishing the Laplacian $\Delta$ from the Gaussian kernel $\Delta_n$, a direct computation gives
$$
  \Delta \Delta_n(\boldsymbol{x}; \Sigma) = \left( \frac{\left\lvert \boldsymbol{x} - \boldsymbol{Y}_n \right\rvert^2}{\Sigma^4} - \frac{3}{\Sigma^2} \right) \Delta_n\left(\boldsymbol{x}; \Sigma\right) \text{,}
$$
and, since $\sigma \leq \Sigma$, we can write
$$
  \tilde{\Delta}_n\left(\boldsymbol{x}; \Sigma\right) = \left(a_0 + a_2 r_n^2 \right) \Delta_n\left(\boldsymbol{x}; \Sigma\right) \text{,}
$$
where $a_0 = 1 - \frac{3\left(\sigma^2 - \Sigma^2\right)}{2 \Sigma^2}$, $a_2 = \frac{\sigma^2 - \Sigma^2}{2\Sigma^4}$ and $r_n = \left\lvert \boldsymbol{x} - \boldsymbol{Y}_n \right\rvert$. Standard FCM is then the case where $a_0 = 1$ and $a_2 = 0$.

The resulting function $\tilde{\mathcal{J}}^\dagger\left[\left(\boldsymbol{F}_n\right)_{n = 1}^N\right]$ is computed on the gridpoints for the Stokes solver, but is truncated to be supported on a stencil of $M_G \times M_G \times M_G$ gridpoints.

In the code, we define
- `σ` $= \sigma$,
- `Σ` $= \Sigma$,
- `num_grid_points` $= (M_x, M_y, M_z)$,
- `M_G` $= M_G$,
- `h` $= h$ — the uniform grid spacing $h = \frac{L_i}{M_i}$, equal across axes,
- `a` $= a = 1$,
- `Σ_over_σ` $= \frac{\Sigma}{\sigma}$, the actual control parameter, see §5 in the paper.

## Method

### The spread on the grid

Sampled on the grid and truncated to each particle's stencil, the spreading operator evaluates to
$$
\tilde{\mathcal{J}}^\dagger[\mathcal{F}](\boldsymbol{x}_g)
= \sum_{n=1}^N \boldsymbol{F}_n \bigl(a_0 + a_2 r_n^2\bigr) \Delta_n(\boldsymbol{x}_g; \Sigma),
\qquad r_n = \lvert \boldsymbol{x}_g - \boldsymbol{Y}_n \rvert,
$$
at every grid point $\boldsymbol{x}_g$, with each particle contributing only to the $M_G^3$ grid points of its stencil. This is the quantity `spread_forces!` writes into `force_density`, using the closed form of $\tilde{\Delta}_n$ derived in the Summary.

### Separability of the Gaussian

With the one-dimensional Gaussian $g(s; \Sigma) = (2\pi\Sigma^2)^{-1/2}\exp(-s^2 / 2\Sigma^2)$, the isotropic three-dimensional Gaussian factors per axis,
$$
\Delta_n(\boldsymbol{x}_g; \Sigma)
= g\bigl(x_{i_x} - Y_{n,1}; \Sigma\bigr)
  g\bigl(y_{i_y} - Y_{n,2}; \Sigma\bigr)
  g\bigl(z_{i_z} - Y_{n,3}; \Sigma\bigr),
$$
so the three-dimensional normalisation $(2\pi\Sigma^2)^{-3/2}$ is the product of three one-dimensional factors $(2\pi\Sigma^2)^{-1/2}$. The code therefore stores a single per-axis `inv_norm = 1/√(2πΣ²)` and multiplies three of them. The squared distance likewise splits per axis,
$$
r_n^2 = (x_{i_x} - Y_{n,1})^2 + (y_{i_y} - Y_{n,2})^2 + (z_{i_z} - Y_{n,3})^2 .
$$
Per particle, the kernel precomputes the per-axis one-dimensional Gaussian weights (`stencil_gaussian`, $\mathcal{O}(M_G)$ exponentials per axis) and the per-axis squared distances (`stencil_r²`); the inner $M_G^3$ loop then assembles $\tilde{\Delta}_n$ with three multiplies and the polynomial factor $(a_0 + a_2 r_n^2)$, evaluating no further exponentials.

### Per-particle stencil — nearest-anchored

For axis $i \in \{1, 2, 3\}$ and particle $n$, the stencil is anchored at the nearest grid point,
$$
j_i = \operatorname{round}\!\bigl(Y_{n,i} / h\bigr)
\qquad \text{(Julia: `round(Int32, Y/h, RoundNearestTiesToEven)`)} .
$$
The stencil grid coordinates on axis $i$ are $x^{(s)}_i = \bigl(j_i - \lfloor M_G/2 \rfloor + s\bigr)h$ for $s = 0, \dots, M_G - 1$, and the corresponding periodic grid index is $\bigl(j_i - \lfloor M_G/2 \rfloor + s\bigr) \bmod M_i$, mapped to 1-based as `mod(⋅, M_i) + 1`. For odd $M_G$ the stencil is symmetric about $j_i$; for even $M_G$ it covers $\lfloor M_G/2 \rfloor$ points below $j_i$ and $\lceil M_G/2 \rceil - 1$ above. The nearest-grid-point anchor is an implementation convention (matching the reference implementation); the $(M_G, \Sigma/h)$ accuracy calibration it serves is given in §5 (Table 1 and Fig. 1(a)).

### The limit $\Sigma = \sigma$

At $\Sigma = \sigma$ the coefficients collapse to $a_0 = 1$, $a_2 = 0$, so $\tilde{\Delta}_n = \Delta_n$ and the spread is identical to the standard-FCM monopole spread at the same grid spacing. No separate code path is needed; the polynomial factor becomes the constant 1.

## Contract

### Cold-path input (user-supplied to `FFCMConfig`)

- `a::T = T(1)` — particle radius. **Must equal `T(1)`** in the current
  implementation; any other value is rejected with an `ArgumentError`. All
  lengths are expressed in units of the particle radius (see the kernel
  discussion above).
- `Σ_over_σ::T` — required; the ratio $\frac{\Sigma}{\sigma}$ (paper §5). Must
  satisfy $\frac{\Sigma}{\sigma} \geq 1$. The equality case $\Sigma = \sigma$
  reduces to standard FCM; the strict inequality $\Sigma > \sigma$ is the regime
  where fast FCM reduces cost. Equality is supported because it is free and useful
  for validation.
- `num_grid_points::NTuple{3, Int32}` — required; the per-axis grid dimensions
  $(M_x, M_y, M_z)$. Each component must be at least 1.
- `M_G::Integer` — required; the cubic stencil width per axis. Stored as `Int32`;
  must be at least 2.

The cold-path inputs from steps 1–2 (`L`, `R_c`, `N`) are unchanged.

### Cold-path derived state (computed once, stored on `FFCMConfig`)

| Field | Type | Definition |
|---|---|---|
| `σ` | `T` | $\sigma = \frac{a}{\sqrt{\pi}}$ (paper §2). |
| `Σ` | `T` | $\Sigma = \frac{\Sigma}{\sigma}\sigma$ (paper §3). |
| `num_grid_points` | `NTuple{3, Int32}` | The user input $(M_x, M_y, M_z)$. |
| `M_G` | `Int32` | The user input. |
| `h` | `T` | Uniform grid spacing $\frac{L_i}{M_i}$, equal across axes (validated). |
| `inv_h` | `T` | Precomputed $\frac{1}{h}$ (the hot path multiplies). |
| `force_density` | `StructArray{SVector{3, T}, 3, …}` | Shape $(M_x, M_y, M_z)$; three-component vector field, struct-of-arrays backed. |
| `stencil_gaussian` | `StructArray{SVector{3, T}, 1, …}` | Length $M_G$; per-particle, per-axis 1-D Gaussian weights. |
| `stencil_r²` | `StructArray{SVector{3, T}, 1, …}` | Length $M_G$; per-particle, per-axis squared distances. |
| `stencil_index` | `StructArray{SVector{3, Int32}, 1, …}` | Length $M_G$; per-particle, per-axis 1-based periodic-wrapped stencil indices. |

Cold-path validation (constructor): `a == T(1)`; $\frac{\Sigma}{\sigma} \geq 1$;
$M_G \geq 2$; all `num_grid_points[i] ≥ 1`; $M_G \leq \min(M_x, M_y, M_z)$;
isotropic spacing
$\frac{L_x}{M_x} = \frac{L_y}{M_y} = \frac{L_z}{M_z}$ within $\sqrt{\mathrm{eps}(T)}$
relative tolerance. The $M_G \leq \min(M_x, M_y, M_z)$ bound is what makes the
inner-loop `@simd` argument (Performance notes) correct: a stencil wider than the
grid on some axis would wrap distinct stencil points onto the same grid point.

### Hot-path input (per `mobility!` call, populated by steps 1–2)

- `config.particles.Y_sorted`, `config.particles.F_sorted` (shape $(3, N)$) — sorted positions and
  forces from `sort_particles_by_cell!`.

### Hot-path output

- `config.grid.force_density` populated with $\tilde{\mathcal{J}}^\dagger[\mathcal{F}](\boldsymbol{x}_g)$
  at every grid point $\boldsymbol{x}_g = ((i_x - 1)h, (i_y - 1)h, (i_z - 1)h)$
  for $i_x \in 1{:}M_x$, $i_y \in 1{:}M_y$, $i_z \in 1{:}M_z$. The grid is zeroed
  at the start of the call.

The scratch fields (`stencil_gaussian`, `stencil_r²`, `stencil_index`) are overwritten
with the last particle's per-axis values; they carry no between-call invariant.

### Periodicity contract

Positions enter `spread_forces!` after `wrap_positions!` (step 1), so
$Y_{n,i} \in [0, L_i)$. Each particle's stencil is wrapped $\bmod\ M_i$ inside the
kernel, so the resulting grid indices lie in $1{:}M_i$. A particle near the domain
edge spreads to a stencil that wraps to the opposite side, identically to an
interior particle.

### Boundary cases

- **Particle exactly on a grid point** (after wrap): distance 0 along each axis;
  no special case.
- **Particle exactly on a cell midpoint** ($Y/h = j + 0.5$): the nearest-anchored
  stencil tie-breaks via `RoundNearestTiesToEven`. Either neighbouring stencil is
  accepted; tests do not pin which.
- **$\Sigma = \sigma$**: $a_0 = 1$, $a_2 = 0$, so $\tilde{\Delta}_n = \Delta_n$
  and the spread reduces to standard FCM.

## Implementation

`spread_forces!(config)` is the hot-path entry point. It reads the sorted
positions and forces `config.particles.Y_sorted`, `config.particles.F_sorted`, writes the spread force
field into `config.grid.force_density` (zeroed at the start of the call), and returns
`config`. It allocates nothing and is type-stable, so it can run on every
iteration of a downstream solve. The grid and the per-particle scratch buffers are
built once on the cold path, by the `FFCMConfig` constructor (its inputs and
validation are listed under Contract).

The hot path delegates to two function-barrier kernels that take naked arrays and
scalars rather than the `config`. The barrier gives the compiler a concrete-typed
call to specialise (which keeps the loop type-stable and allocation-free) and lets
each kernel be exercised in isolation by the tests:

- `_spread_forces_kernel!` zeroes the three grid components, computes the scalar
  coefficients $a_0$ and $a_2$ once, then loops over the sorted particles,
  accumulating $\boldsymbol{F}_n(a_0 + a_2 r_n^2)g_x g_y g_z$ into the
  components of `force_density`.
- `_fill_particle_stencil!` computes, for one particle, the per-axis Gaussian
  weights `stencil_gaussian`, the axis-squared distances `stencil_r²`, and the
  periodic-wrapped 1-based indices `stencil_index` of its $M_G^3$ stencil. It is shared
  verbatim with interpolation ([interpolation.md](interpolation.md)): interpolation
  is the exact discrete adjoint of the spread, so both must place and weight the
  stencil identically, and keeping the convention in one function is what
  guarantees that.

### Cold-path vs hot-path

| Phase | Allocations | Functions |
|---|---|---|
| Cold | OK | `FFCMConfig` constructor: validate inputs; derive `σ, Σ, h, inv_h`; allocate `force_density` and the stencil scratch fields. |
| Hot  | `@ballocated == 0` | `spread_forces!(config)` (after wrap → assign → sort). |

The hot path is allocation-free and type-stable on `T <: AbstractFloat`.

### Storage choice

`force_density` is a `StructArray{SVector{3, T}, 3, …}` backed by three contiguous
`Array{T, 3}` of shape $(M_x, M_y, M_z)$ — a vector field with per-grid-point
readability, stored struct-of-arrays per component. The spread kernel
destructures `(fx, fy, fz) = StructArrays.components(force_density)` without
allocating and writes plain arrays; downstream consumers (steps 4–5) read
`force_density[i_x, i_y, i_z]` as an `SVector{3, T}`.

## Performance notes

- **Per-particle precompute.** Three 1-D Gaussian weight vectors, three
  axis-squared-distance vectors, and three integer index vectors,
  $\mathcal{O}(M_G)$ each, are built before the inner $M_G^3$ loop. Inside the
  loop only multiplies, an add, and a fused polynomial multiply run — no
  exponentials, no divisions.
- **Polynomial coefficients.** $a_0$ and $a_2$ are scalars recomputed per call
  from `σ` and `Σ`, which keeps the field set purely physical. The intermediate
  $\sigma^2 - \Sigma^2$ is the code's `σ²_minus_Σ²`.
- **Normalisation.** The 1-D norm $\frac{1}{\sqrt{2\pi\Sigma^2}}$ is computed once
  per call; the 3-D norm $(2\pi\Sigma^2)^{-3/2}$ is the product of three 1-D
  weights.
- **Inner-loop `@simd`, serial particle loop.** The innermost $k_x$ loop carries
  `@simd`: within one particle's stencil the wrapped indices $\bmod\ M_x$ are
  distinct, so the writes do not alias — guaranteed by the cold-path precondition
  $M_G \leq \min(M_x, M_y, M_z)$ (see the validation note in Contract). The outer
  particle loop is
  `@inbounds` only: two different particles can write the same grid point, so it
  is neither vectorised nor threaded in the MVP. The sorted-by-cell order from
  step 2 gives spatial locality on consecutive particles' stencil writes.
- **Buffer zeroing.** `fill!` each of `fx, fy, fz` once per call before the
  particle loop — non-allocating, amortised over $N \cdot M_G^3$ work.
- **Concurrency.** Under threading or on the GPU the scatter needs atomic adds or
  thread-local accumulators; this is a concern for the future parallel backends.

### Future optimisation — Fourier-space split

A mathematically equivalent alternative spreads only the plain Gaussian
$\boldsymbol{F}_n \Delta_n$ and folds the polynomial factor into the Stokes solve
as a Fourier-space multiply by $\bigl(1 + \frac{\Sigma^2 - \sigma^2}{2} k^2\bigr)^2$
(squared because spread and interpolation both carry the factor). This trades the
per-particle $\mathcal{O}(M_G^3)$ polynomial multiply for an $\mathcal{O}(M)$
Fourier-space multiply in step 4. Not implemented; recorded for future evaluation.

## Verification

Test files follow the flat-`test/` layout and the domain-language naming
convention.

- `test/test_spread_forces.jl` — analytical and paper-derived tests:
  1. **Cold-path validation.** `a ≠ T(1)` is rejected; `Σ_over_σ < T(1)` is
     rejected; anisotropic spacing is rejected. Derived fields satisfy the closed
     forms ($\sigma = \frac{1}{\sqrt{\pi}}$, $\Sigma = \frac{\Sigma}{\sigma}\sigma$,
     $h = \frac{L_1}{M_x}$). `force_density` has shape $(M_x, M_y, M_z)$; the stencil
     scratch fields have length $M_G$.
  2. **Closed-form single-particle stencil.** One particle at a known position;
     hand-computed $\tilde{\Delta}_n(\boldsymbol{x}_g; \Sigma)$ at stencil points
     matches `force_density` to `sqrt(eps(T))`.
  3. **Stencil anchoring.** For $Y = 0.3h$ (even $M_G$) the lowest occupied
     index is $\operatorname{round}(Y/h) - M_G/2$; for $Y = 0.7h$ it is one
     higher (after wrap).
  4. **Force conservation.** $\sum_g \tilde{\mathcal{J}}^\dagger[\mathcal{F}](\boldsymbol{x}_g) h^3 \approx \sum_n \boldsymbol{F}_n$,
     within the $(M_G, \Sigma/h)$ truncation tolerance (paper Table 1). This
     holds because $\int_{\mathbb{R}^3} \tilde{\Delta}_n \, \mathrm{d}^3\boldsymbol{x} = 1$:
     the Gaussian integrates to one and the Laplacian term integrates to zero
     ($\int \Delta\Delta_n = 0$ by the decay of $\nabla\Delta_n$), so only the
     $M_G^3$-stencil truncation of the Gaussian tail breaks the equality.
  5. **First moment / centroid.** $\sum_g \boldsymbol{x}_g \tilde{\Delta}_n(\boldsymbol{x}_g; \Sigma) h^3 \approx \boldsymbol{Y}_n$,
     same tolerance, because $\int_{\mathbb{R}^3} \boldsymbol{x} \tilde{\Delta}_n \, \mathrm{d}^3\boldsymbol{x} = \boldsymbol{Y}_n$
     (the Gaussian has mean $\boldsymbol{Y}_n$ and $\int \boldsymbol{x}\Delta\Delta_n = 0$).
  6. **Periodicity.** $\boldsymbol{Y}_n$ versus $\boldsymbol{Y}_n + L\hat{e}_i$ →
     identical `force_density`.
  7. **Linearity in `F`.** $\tilde{\mathcal{J}}^\dagger[\alpha\mathcal{F}_1 + \beta\mathcal{F}_2] = \alpha\tilde{\mathcal{J}}^\dagger[\mathcal{F}_1] + \beta\tilde{\mathcal{J}}^\dagger[\mathcal{F}_2]$.
  8. **Reflection symmetry.** Particle on a grid point → `force_density` is
     reflection-symmetric about it per axis, within `sqrt(eps(T))`, because the
     Gaussian is even in $\boldsymbol{x} - \boldsymbol{Y}_n$.
  9. **Translation by one grid step.** $\boldsymbol{Y}_n + h\hat{e}_i$ →
     `force_density` shifted by one cell on axis $i$.
  10. **Standard FCM at $\Sigma = \sigma$.** Equals the closed-form standard-FCM
      Gaussian spread at every grid point.
- `test/test_spread_forces_inferred.jl` — `@inferred` for `spread_forces!` and the
  kernels, `Float32`/`Float64`.
- `test/test_spread_forces_allocations.jl` — `@ballocated == 0` for the wrapper
  and the kernels.
- `test/test_jet.jl` — `JET.@test_call spread_forces!` over the full call graph.
- `test/test_aqua.jl` — `Aqua.test_all` package hygiene.
- `test/test_fcm_grid.jl` — pins the cold-path derived fields.

Tolerances: tests 1–3 and 6–10 use `sqrt(eps(T))`; the truncation-dominated tests
4–5 use the paper-tabulated error for the chosen $(M_G, \Sigma/h)$, documented in
the test body.

## Differences from cuFCM

> Comparison against the C++/CUDA reference implementation, kept for validation
> during development and removed once the port is complete.

Audited: `cuFCM/src/CUFCM_FCM.cu`
(`cufcm_mono_dipole_distribution_bpp_shared_dynamic`), `CUFCM_FCM.cuh`,
`CUFCM_SOLVER.cu`. Only the active (uncommented) kernel variant is considered; the
`_tpp_register`, `_recompute`, `_selection`, `_mono` variants are dead code per
the cuFCM convention.

| Facet | cuFCM | This package | Decision |
|---|---|---|---|
| Grid memory layout | SoA per component: three linear `myCufftReal*` buffers `hx, hy, hz`, `ind = ix + iy⋅nx + iz⋅nx⋅ny`. | `force_density::StructArray{SVector{3, T}, 3}` over three `Array{T, 3}`. | **Adopt** SoA, wrapped for a per-grid-point `SVector{3, T}` API; byte-compatible with cuFCM for parity, stride-1 along $i_x$. |
| Stencil anchoring | `xg = my_rint(Y/dx) − ngdh + (i mod ngd)`, `ngdh = ngd/2`. | Identical: $j_i = \operatorname{round}(Y_{n,i}/h)$, stencil $j_i - \lfloor M_G/2\rfloor + s$. | **Adopt** — nearest-anchoring minimises truncation for fixed $M_G$. |
| Periodic wrap | `xg − nx⋅floor(xg/nx)`. | `mod(j_x − ⌊M_G/2⌋ + s, M_x)`. | **Keep** — semantically identical. |
| Normalisation | `Anorm = 1/√(2π⋅Σ²)` per axis; 3-D as `Anorm³`. | Identical (`inv_norm`). | **Keep**. |
| Polynomial coefficients | `temp2 = ½⋅pdmag/Σ²`, `temp3 = temp2/Σ²`, `temp4 = 3⋅temp2`, `pdmag = σ²−Σ²`; factor `(1 + temp3⋅r² − temp4)`. | $a_0 = 1 − \frac{3(\sigma^2-\Sigma^2)}{2\Sigma^2}$, $a_2 = \frac{\sigma^2-\Sigma^2}{2\Sigma^4}$. | **Keep** — algebraically identical; named after the closed form. |
| Per-particle precompute | shared-mem `gaussx/y/z`, `xdis/ydis/zdis`, `indx/y/z`, grad-Gaussian (rotation), scalars. | `stencil_gaussian`, `stencil_r²`, `stencil_index`, length $M_G$. Stores $r^2_i = x_i^2$ rather than signed `xdis`. | **Adopt** the pattern; store $r^2$ directly (one fewer multiply per inner iteration). |
| Scatter into grid | `atomicAdd(&fx[ind], …)` for the many-to-one race. | Plain `fx[i_x, i_y, i_z] += …` (single-threaded CPU). | **Keep** for the MVP; threading needs atomics or thread-local accumulators. |
| Particle iteration | one CUDA block per particle; `Y[3⋅np + k]` from raw arrays (the sort index is a filter, not an indirection). | serial loop over sorted slots; reads materialised `Y_sorted`/`F_sorted`. | **Keep** — step 2 paid the gather; consecutive sorted particles give cache locality. |
| Dipole / torque / rotation | `rotation == 1` branch spreads $\boldsymbol{H}\nabla\Delta$. | none — force-only $\mathcal{M}^{\mathcal{V}\mathcal{F}}$. | **Out of scope** for the force-only operator. |
| `USE_REGULARFCM` mode | compile-time branch with $\Sigma = \sigma$, skips the polynomial. | no flag; $\frac{\Sigma}{\sigma} = 1$ collapses the polynomial naturally. | **Subsume via the $\Sigma = \sigma$ limit** (test 10). |
| Isotropic `dx` | scalar `Real dx`. | scalar `h`; validated isotropic at construction. | **Keep**. |
| Rounding | `my_rint` (ties to even). | `round(…, RoundNearestTiesToEven)`. | **Keep** — parity except at exact ties (tests do not pin which). |

The behavioural differences with implementation consequence are the SoA storage
via `StructArray`, the $r^2$-only precompute, and the absent threading and
rotation branches.
