# Force Spreading

Step 3 of the Fast FCM algorithm (Su & Keaveny 2024, §4 Step 3;
`paper/tex/outline.tex:520`), defined operationally by reference to §3
Step `spread` (`paper/tex/outline.tex:178`) with the FCM operator
$\mathcal{J}$ replaced by the modified-kernel spreading operator
$\widetilde{\mathcal{J}}^\dagger$ and the Gaussian kernel
$\Delta_n(\bm{x}; \sigma)$ replaced by the modified kernel
$\widetilde{\Delta}_n(\bm{x}; \Sigma)$.

Given particle positions and forces gathered into sorted order by step 2
(see [particle-sorting.md](particle-sorting.md)), step 3 evaluates the
modified-kernel-weighted force density on a uniform periodic Cartesian
grid:

$$
\widetilde{\mathcal{J}}^\dagger[\mathcal{F}](\bm{x}_g)
= \sum_{n=1}^{N} \bm{F}_n\, \widetilde{\Delta}_n(\bm{x}_g; \Sigma)
\quad \text{for every grid point } \bm{x}_g .
$$

This grid force field is the input to step 4 (Stokes solve). The grid
layout and storage choice committed in this document pin that interface.

## Notation (paper-consistent)

Following paper notation conventions (paper §2 onward; CLAUDE.local.md
*Notation conventions*):

- `i ∈ {x, y, z}` (equivalently `{1, 2, 3}`) — Cartesian axis index.
- `n ∈ {1, …, N}` — particle index.
- `Y[i, n]` is the `i`-th component of particle `n`'s position; `F[i, n]`
  likewise for force. Paper symbols `Y_n`, `F_n` map to the `n`-th column.
- `(M_x, M_y, M_z)` (paper §3, `outline.tex:175`) — number of grid points
  per axis. Bundled in code as `num_grid_points::NTuple{3, Int32}` (long
  form per the convention that the paper does not bundle them under a
  single symbol).
- `M_G` (paper §5, `outline.tex:571`) — kernel grid-support per axis. The
  paper-supplied stencil is $M_G \times M_G \times M_G$.
- `Δx` (paper §3, `outline.tex:175`) — uniform grid spacing.
- `a` (paper §2, `outline.tex:163`) — particle hydrodynamic radius.
- `σ` (paper §2, eq 148 / `outline.tex:147-149`) — FCM Gaussian envelope.
  Derived from `a` by $a = \sigma\sqrt{\pi}$.
- `Σ` (paper §3, eq 267 / `outline.tex:265-269`) — fast-FCM modified-kernel
  width. Constrained by $\Sigma \geq \sigma$.
- `Σ/σ` (paper §5, `outline.tex:573`) — the fast-FCM resolution control
  parameter; user-supplied via `Σ_over_σ`.

## Contract

### Cold-path input (user-supplied to `FFCMConfig`)

- `a::T = T(1)` — particle radius. **Must equal `T(1)` in the current
  implementation.** Any other value triggers
  `error("non-unit particle radius not yet implemented")`. The parameter
  is exposed as a documented contract placeholder for a future
  generalisation. The CLAUDE.md project contract sets the unit-radius
  convention.
- `Σ_over_σ::T` — required; the ratio $\Sigma/\sigma$ from paper §5
  (`outline.tex:573`). Must satisfy $\Sigma_\mathrm{over\_\sigma} \geq T(1)$;
  the equality case $\Sigma = \sigma$ is the standard-FCM degenerate
  limit (see *Standard FCM via degenerate limit* below). The paper's
  strict inequality $\Sigma > \sigma$ (eq 269) is the regime where the
  fast-FCM cost reduction applies; equality is supported because it is
  algorithmically free and useful for validation.
- `num_grid_points::NTuple{3, Int32}` — required; the per-axis grid
  dimensions $(M_x, M_y, M_z)$. Each component must be at least 1.
- `M_G::Integer` — required; the kernel stencil width per axis. Stored
  as `Int32`. Must be at least 2.

The existing cold-path inputs from steps 1 and 2 (`L`, `R_c`, `N`) are
unchanged.

### Cold-path derived state (computed once, stored on `FFCMConfig`)

| Field | Type | Definition |
|---|---|---|
| `σ` | `T` | $\sigma = a / \sqrt{\pi}$ (paper §2, `outline.tex:163`). |
| `Σ` | `T` | $\Sigma = (\Sigma/\sigma) \cdot \sigma$ (paper §3, eq 267). |
| `num_grid_points` | `NTuple{3, Int32}` | The user input. |
| `M_G` | `Int32` | The user input. |
| `Δx` | `T` | Uniform grid spacing $L_i / M_i$, equal across axes (validated). |
| `inv_Δx` | `T` | Precomputed $1 / \Delta x$ (hot path multiplies). |
| `force_grid` | `StructArray{SVector{3, T}, 3, …}` | Shape `(M_x, M_y, M_z)`; SoA-backed three-component vector field. See *Storage choice*. |
| `gauss_x`, `gauss_y`, `gauss_z` | `Vector{T}` | Length `M_G`; per-particle 1-D Gaussian weights, written every spread call. |
| `r²_x`, `r²_y`, `r²_z` | `Vector{T}` | Length `M_G`; per-particle axis-squared distances. |
| `ind_x`, `ind_y`, `ind_z` | `Vector{Int32}` | Length `M_G`; per-particle 1-based stencil grid indices (after periodic wrap). |

Cold-path validation (constructor):

- `a == T(1)` else `error("non-unit particle radius not yet implemented")`.
- `Σ_over_σ ≥ T(1)` else `ArgumentError("Σ/σ must be at least 1; got ...")`.
- `M_G ≥ 2` else `ArgumentError`.
- All `num_grid_points[i] ≥ 1` else `ArgumentError`.
- $L_x/M_x = L_y/M_y = L_z/M_z$ within $\sqrt{\mathrm{eps}(T)}$ relative
  tolerance else `ArgumentError("anisotropic grid spacing not supported")`.

The existing validation from steps 1 and 2 (`L > 0`, `R_c > 0`, `N > 0`)
is retained.

### Hot-path input (per `mobility!` call, populated by steps 1 and 2)

- `config.Y_sorted::Matrix{T}`, `config.F_sorted::Matrix{T}` (shape
  `(3, N)`) — sorted positions and forces from
  `sort_particles_by_cell!`.

### Hot-path output

- `config.force_grid` populated with
  $\widetilde{\mathcal{J}}^\dagger[\mathcal{F}](\bm{x}_g)$ at every grid
  point $\bm{x}_g = ((i_x-1)\Delta x, (i_y-1)\Delta x, (i_z-1)\Delta x)$
  for $i_x \in 1{:}M_x$, $i_y \in 1{:}M_y$, $i_z \in 1{:}M_z$.

The scratch buffers (`gauss_*`, `r²_*`, `ind_*`) are overwritten with
the last particle's per-axis precomputed values; they carry no
between-call invariant.

### Periodicity contract

Particle positions enter `spread_forces!` after `wrap_positions!` from
step 1, so $Y^n_i \in [0, L_i)$ is guaranteed. The stencil for each
particle is wrapped mod $M_i$ inside the kernel; the resulting grid
indices lie in $1{:}M_i$ for each axis. A particle near the domain
edge spreads to a stencil that wraps to the opposite side, identically
to a particle in the interior.

### Boundary cases

- **Particle exactly on a grid point** (after wrap). The closed-form
  $\widetilde{\Delta}_n$ is evaluated at distance 0 along each axis with
  weight $a_0 \cdot (2\pi\Sigma^2)^{-3/2}$. No special case.
- **Particle exactly on a cell midpoint** (`Y/Δx = j + 0.5`). The
  nearest-anchored stencil tie-breaks via `RoundNearestTiesToEven`
  (Julia default, matching cuFCM's `my_rint`). Either neighbouring
  stencil is accepted; tests do not pin which one.
- **`Σ = σ` degenerate limit.** $a_0 = 1$, $a_2 = 0$, so
  $\widetilde{\Delta}_n = \Delta_n$. The polynomial multiplication runs
  but produces a no-op factor.

## API

```julia
"""
    FFCMConfig{T}(; L, R_c, N, a = T(1), Σ_over_σ, num_grid_points, M_G)

Cold-path configuration of the Fast FCM mobility operator. Owns the cell
geometry (steps 1, 2) and the FCM grid (this step), plus all hot-path
buffers sized for `N` particles and the chosen grid. Built once, reused
across many `mobility!` calls.

Keyword arguments specific to step 3:

- `a::T = T(1)` — particle hydrodynamic radius. Only `a == T(1)` is
  currently supported (paper §2; CLAUDE.md unit-radius convention).
- `Σ_over_σ::T` — kernel resolution ratio Σ/σ (paper §5,
  `outline.tex:573`); must satisfy `Σ_over_σ ≥ T(1)`. The equality case
  is the standard-FCM degenerate limit.
- `num_grid_points::NTuple{3, Int32}` — grid dimensions (M_x, M_y, M_z)
  (paper §3, `outline.tex:175`).
- `M_G::Integer` — cubic stencil support per axis (paper §5,
  `outline.tex:571`). Stored as `Int32`.

The grid spacing `Δx = L_i / num_grid_points[i]` is required to be
identical across axes (paper §3 isotropy assumption).

See `spec/force-spreading.md`.
"""
struct FFCMConfig{T <: AbstractFloat}
    # … (existing fields from steps 1, 2) …
    a::T
    σ::T
    Σ::T
    num_grid_points::NTuple{3, Int32}
    M_G::Int32
    Δx::T
    inv_Δx::T
    force_grid::StructArray{SVector{3, T}, 3, /*…*/}
    gauss_x::Vector{T}; gauss_y::Vector{T}; gauss_z::Vector{T}
    r²_x::Vector{T};   r²_y::Vector{T};   r²_z::Vector{T}
    ind_x::Vector{Int32}; ind_y::Vector{Int32}; ind_z::Vector{Int32}
end

"""
    spread_forces!(config) -> config

Step 3 of the Fast FCM algorithm (Su & Keaveny 2024, §4 Step 3). Evaluate
J̃†[F](x_g) = Σₙ Fₙ Δ̃ₙ(x_g; Σ) on every grid point x_g, writing the result
into `config.force_grid`. Reads `config.Y_sorted` and `config.F_sorted`
(populated by `sort_particles_by_cell!`).

Allocation-free and type-stable on `T <: AbstractFloat`.

See `spec/force-spreading.md`.
"""
function spread_forces!(config::FFCMConfig{T}) where {T} end
```

The underscore-prefixed
`_spread_forces_kernel!(force_grid, Y_sorted, F_sorted, σ, Σ, Δx, inv_Δx,
num_grid_points, M_G, gauss_x, gauss_y, gauss_z, r²_x, r²_y, r²_z,
ind_x, ind_y, ind_z)` is the function-barrier kernel: it takes naked
arrays and scalars so it is independently testable and benefits from
Julia's standard type-stability pattern.

## Modified kernel — closed form

From paper eq 267 (`outline.tex:265-269`):

$$
\widetilde{\Delta}_n(\bm{x}; \Sigma)
= \left(1 + \frac{\sigma^2 - \Sigma^2}{2}\, \nabla^2\right) \Delta_n(\bm{x}; \Sigma).
$$

The Laplacian of the FCM Gaussian (paper eq 148, `outline.tex:147-149`)
is itself a polynomial-in-$r^2$ times the same Gaussian. With
$r = |\bm{x} - \bm{Y}_n|$,
$$
\nabla^2 \Delta_n(\bm{x}; \Sigma)
= \left(\frac{r^2}{\Sigma^4} - \frac{3}{\Sigma^2}\right)\, \Delta_n(\bm{x}; \Sigma).
$$
Therefore, with $\mathrm{pdmag} = \sigma^2 - \Sigma^2 \leq 0$,
$$
\widetilde{\Delta}_n(\bm{x}; \Sigma)
= \bigl(a_0 + a_2\, r^2\bigr)\, \Delta_n(\bm{x}; \Sigma),
\qquad a_0 = 1 - \frac{3\, \mathrm{pdmag}}{2\Sigma^2},
\qquad a_2 = \frac{\mathrm{pdmag}}{2\Sigma^4}.
$$
At $\Sigma = \sigma$ this reduces to $a_0 = 1$, $a_2 = 0$ and
$\widetilde{\Delta}_n = \Delta_n$ — the standard FCM monopole kernel.

### Separability of the Gaussian

With $g(s; \Sigma) = (2\pi\Sigma^2)^{-1/2}\exp(-s^2 / 2\Sigma^2)$ the 1-D
Gaussian, the 3-D Gaussian factors per axis:
$$
\Delta_n(\bm{x}_g; \Sigma)
= g(x_{i_x} - Y_{n,1};\Sigma)\, g(y_{i_y} - Y_{n,2};\Sigma)\, g(z_{i_z} - Y_{n,3};\Sigma).
$$
Per particle, the three 1-D weight vectors `gauss_x[1..M_G]`,
`gauss_y[1..M_G]`, `gauss_z[1..M_G]` are precomputed at $\mathcal{O}(M_G)$
exponentials each; the inner $M_G^3$ stencil loop then assembles
$\widetilde{\Delta}_n$ at each grid point with three multiplies plus a
polynomial term, no further exponentials.

The polynomial $r^2 = (x - Y_{n,1})^2 + (y - Y_{n,2})^2 + (z - Y_{n,3})^2$
likewise sums precomputed axis-squared distances
`r²_x[..] + r²_y[..] + r²_z[..]`.

### Per-particle stencil — nearest-anchored

For axis $i \in \{1, 2, 3\}$ and particle $n$,
$$
j_i = \mathrm{round}\!\bigl(Y_{n,i} \cdot \mathrm{inv}\Delta x\bigr)
\qquad \text{(Julia: `round(Int32, ..., RoundNearestTiesToEven)`)}.
$$
The stencil grid coordinates on axis $i$ are
$x_{i_x}^{(s)} = (j_i - \lfloor M_G/2\rfloor + s) \cdot \Delta x$ for
$s = 0, \ldots, M_G - 1$. The corresponding grid index is
$(j_i - \lfloor M_G/2\rfloor + s) \bmod M_i$, mapped to 1-based for
Julia indexing (`mod(...) + 1`).

For odd $M_G$ the stencil is symmetric around $j_i$; for even $M_G$ it
covers $\lfloor M_G/2\rfloor$ grid points below $j_i$ and
$\lceil M_G/2\rceil - 1$ above (cuFCM convention).

### Standard FCM via degenerate limit

Setting `Σ_over_σ = T(1)` collapses the modified kernel to the standard
FCM Gaussian: $a_0 = 1$, $a_2 = 0$, $\widetilde{\Delta}_n = \Delta_n$.
The spread is then bit-for-bit equivalent to a standard-FCM spread at
the same grid spacing. This makes step 3 a drop-in replacement for the
standard-FCM monopole spread when the user opts out of the fast-FCM
correction; no separate code path is needed (compare cuFCM's
`USE_REGULARFCM` compile-time switch in `cuFCM/src/CUFCM_FCM.cu:33-67`).

## Cold-path vs hot-path

| Phase | Allocations | Functions |
|---|---|---|
| Cold | OK | `FFCMConfig` constructor: validate inputs; derive `σ, Σ, Δx, inv_Δx`; allocate `force_grid` and the per-axis scratch vectors. |
| Hot  | `@ballocated == 0` | `spread_forces!(config)` (after `wrap_positions!` → `assign_cells!` → `sort_particles_by_cell!`). |

The hot path is allocation-free and type-stable on `T <: AbstractFloat`.

## Performance notes

- **Storage choice.** `force_grid` is a
  `StructArray{SVector{3, T}, 3, …}` backed by three contiguous
  `Array{T, 3}` of shape `(M_x, M_y, M_z)` — semantically a vector
  field with per-grid-point readability, physically SoA per component.
  The spread kernel destructures the components via
  `(fx, fy, fz) = StructArrays.components(force_grid)` (zero allocation —
  just unpacks the backing tuple) and writes them as plain arrays.
  Downstream consumers (step 4 Stokes solve, step 5 interpolation) read
  `force_grid[i_x, i_y, i_z]` as an `SVector{3, T}` directly.
- **SIMD on the inner loop.** With SoA storage, `fx[i_x, i_y, i_z]` walks
  stride-1 along $i_x$; the innermost stencil loop can therefore vectorise
  cleanly when the wrapped indices are sequential (the common interior
  case). The vectorisation is verified post-implementation by benchmark,
  per the `/julia-numerical-computing` guidance against pre-emptive
  `@simd`.
- **Per-particle precompute.** Three 1-D Gaussian weight vectors, three
  axis-squared-distance vectors, and three integer index vectors,
  $\mathcal{O}(M_G)$ each, are built before the inner $M_G^3$ stencil
  loop. Inside the loop only multiplies, an add, and a fused polynomial
  multiplication run — no exponentials, no divisions.
- **Polynomial coefficients.** $a_0$ and $a_2$ are scalar, recomputed
  per call from `σ` and `Σ`. Recomputing — rather than storing them as
  cold-path fields — keeps the field set purely physical (`a, σ, Σ`)
  and costs nothing.
- **Normalisation.** The 1-D Gaussian norm
  $1/\sqrt{2\pi\Sigma^2}$ is computed once per call and multiplied into
  the 1-D weights; the 3-D normalisation
  $(2\pi\Sigma^2)^{-3/2}$ falls out as the product of three 1-D weights.
- **Scatter dependency.** The outer particle loop is `@inbounds` only —
  no `@simd`, no threading on the MVP — because two different particles
  can write to the same grid point. The sorted-by-cell order from step 2
  gives spatial locality on `force_grid` writes (consecutive particles
  spread to overlapping stencils), realising the cache benefit of
  step 2.
- **Buffer zeroing.** `fill!` each of `fx, fy, fz` once per call, before
  the particle loop. This is non-allocating and amortised over $N \cdot M_G^3$
  work.

### Future optimisation — Fourier-space split (out of scope here)

A mathematically equivalent alternative spreads only the plain Gaussian
$\bm{F}_n \Delta_n$ and folds the polynomial factor into the Stokes solve
as a multiplication by $(1 + ((\Sigma^2 - \sigma^2)/2) k^2)^2$ in Fourier
space (raised to the power 2 because spread and interpolation both carry
the factor). This trades the per-particle $O(M_G^3)$ polynomial multiply
for an $O(M_x M_y M_z)$ Fourier-space multiply in step 4 and exposes the
modified-kernel parameter to step 4 rather than localising it here. Not
implemented; documented for future evaluation.

## Diffs from cuFCM

Audited cuFCM files: `cuFCM/src/CUFCM_FCM.cu`
(`cufcm_mono_dipole_distribution_bpp_shared_dynamic`, lines 16–184),
`cuFCM/src/CUFCM_FCM.cuh:48-55`, `cuFCM/src/CUFCM_SOLVER.cu:581-599`.
Only the active (uncommented) kernel variant is considered; the earlier
`_tpp_register`, `_recompute`, `_selection`, and `_mono` variants in the
header are dead code per the cuFCM convention.

| Facet | cuFCM | This package | Decision |
|---|---|---|---|
| Grid memory layout | SoA per component: three `myCufftReal*` buffers `hx, hy, hz`, each linear `(nx·ny·nz,)`, indexed as `ind = ix + iy·nx + iz·nx·ny`. | `force_grid::StructArray{SVector{3, T}, 3, …}` of shape `(M_x, M_y, M_z)`, backed SoA by three `Array{T, 3}` per component (`fx, fy, fz`). | **Adopt** the SoA layout, wrapped in a `StructArray` for a per-grid-point `SVector{3, T}` read API. Matches cuFCM byte-for-byte for CPU↔CUDA parity, lets each component drive an independent r2c FFT plan (cuFCM's pattern), keeps the spread inner loop stride-1 along $i_x$, and provides clean grid-point reads for the Stokes solve. |
| Stencil anchoring | `xg = my_rint(Y/dx) - ngdh + (i mod ngd)` — nearest-grid-point anchor with `ngdh = ngd/2` (integer division). | Identical: $j_i = \mathrm{round}(Y_{n,i}\cdot \mathrm{inv}\Delta x)$, stencil $j_i - \lfloor M_G/2\rfloor + s$ for $s \in 0{:}M_G-1$. | **Adopt** (revised from an earlier floor-anchored draft). Nearest-anchoring minimises truncation error for fixed $M_G$ and matches the convention behind paper Table 2 (`outline.tex:600-612`). |
| Periodic wrap of stencil indices | `xg - nx * floor(xg / nx)` — floor-mod inside the kernel after the anchor. | `mod(j_x - M_G÷2 + s, M_x)` (Julia Euclidean mod). | **Keep** — semantically identical. |
| Normalisation `Anorm` | `1/√(2π·Σ²)` per axis (1-D); the 3-D norm `Anorm^3` appears as a product of three per-axis weights. | Identical. | **Keep**. |
| Polynomial coefficients | `temp2 = 0.5·pdmag/Σ²`, `temp3 = temp2/Σ²`, `temp4 = 3·temp2`, with `pdmag = σ² - Σ²` (≤ 0). Combined factor `temp1·(1 + temp3·r² - temp4)`. | $a_0 = 1 - 3\,\mathrm{pdmag}/(2\Sigma^2)$, $a_2 = \mathrm{pdmag}/(2\Sigma^4)$, $\widetilde\Delta_n = (a_0 + a_2 r^2)\Delta_n$. | **Keep** — algebraically identical. Naming follows the closed-form derivation rather than intermediate temps. |
| Per-particle pre-compute | CUDA shared memory of size `(3·ngd·sizeof(Integer) + 9·ngd·sizeof(Real) + 15·sizeof(Real))` holding `gaussx/y/z`, `xdis/ydis/zdis`, `indx/y/z`, `grad_gauss…` (rotation), and per-particle scalars. | Pre-allocated `gauss_x/y/z`, `r²_x/y/z`, `ind_x/y/z` on `FFCMConfig`, length `M_G` each. Stores axis-squared distances (`r²_x`) rather than signed axis distances (`xdis`). | **Adopt** the precompute pattern, but store $r^2_i = x_i^2$ directly to save one multiply per inner-loop iteration (the spread polynomial only needs $r^2$). |
| Scatter into grid | `atomicAdd(&fx[ind], …)` for each component — GPU atomics for the many-to-one write race. | Plain `fx[i_x, i_y, i_z] += …` etc. — single-threaded CPU, no race. | **Keep** for the MVP. The future CPU-threaded path will need `Threads.Atomic`, cell-partitioned scheduling, or thread-local accumulators (logged as a PLAN.md backlog entry for downstream CPU optimisation and CUDA design). |
| Particle iteration | `for(np = blockIdx.x; ...; np += gridDim.x)`; one CUDA block per particle. Y/F indexed as `Y[3·np + k]` from the raw input arrays (cuFCM does **not** materialise sorted Y/F — the sort index `particle_index[np]` is a bounds filter, not an indirection). | Single serial loop `s = 1, …, N` over the sorted slot range; reads `Y_sorted[k, s]`, `F_sorted[k, s]` materialised by step 2. | **Keep** — step 2 has already paid the gather cost, and the cache-line wins on `force_grid` writes from consecutive sorted particles are the benefit step 2 was designed to enable. |
| Particle selection / start-end filter | `particle_index[np] ∈ [start, end)` for sub-domain scheduling. | No filter; the full sorted range is processed. | **Out of scope** — sub-domain scheduling not part of the CPU MVP. |
| Dipole / torque / rotation branch | `rotation == 1`: precomputes `grad_gauss…_dip`, antisymmetric `g_shared`, and an extra `tempdip` factor; spreads $\bm H \nabla \Delta$ alongside $\bm F \Delta$. | None — force-only $\mathcal{M}^{\mathcal{V}\mathcal{F}}$. | **Out of scope** per PLAN.md. |
| `USE_REGULARFCM` compile-time mode | Alternative kernel branch with `Sigma = sigma` and a separate `sigmadip`; skips the polynomial-in-$r^2$ factor. | No flag, no separate code path. With `Σ_over_σ = 1` the polynomial collapses to $a_0 = 1$, $a_2 = 0$ and $\widetilde{\Delta} = \Delta$ naturally; the validation explicitly admits the equality case. | **Subsume via degenerate limit.** cuFCM's flag is a benchmarking convenience to skip a multiplication; algorithmically the limit is free. Test 10 pins this. |
| Isotropic `dx` | `Real dx` scalar in the kernel signature. | Scalar `Δx::T`; validated isotropic at construction. | **Keep** — same constraint, same generality. |
| Rounding convention | `my_rint` — C99 round-to-nearest, ties to even. | Julia `round(Int32, ..., RoundNearestTiesToEven)`. | **Keep** — byte-for-byte parity except at exact ties (which the tests do not pin). |

The behavioural differences with implementation consequence are: SoA
storage via `StructArray` (cuFCM has bare SoA; ours wraps it for a
typed per-grid-point API), $r^2$-only precompute (slight cost saving),
and the absent threading / rotation / start-end-filter branches (out of
scope).

## Verification

End-to-end correctness for this step is established by the test suite.
Test files follow the flat-`test/` layout and the domain-language naming
convention.

- `test/test_spread_forces.jl` — analytical / paper-derived tests:
  1. **Cold-path validation.** `a == T(1)` constructs; `a ≠ T(1)` raises
     `"non-unit particle radius not yet implemented"`. `Σ_over_σ < T(1)`
     raises `ArgumentError`. Anisotropic grid spacing raises
     `ArgumentError`. Derived fields satisfy the closed forms
     ($\sigma = 1/\sqrt{\pi}$, $\Sigma = (\Sigma/\sigma)\sigma$,
     $\Delta x = L_1/M_x$). `force_grid` has shape `(M_x, M_y, M_z)` and
     `StructArrays.components(force_grid)` is a 3-tuple of `Array{T, 3}`
     of that shape; scratch vectors have length `M_G`.
  2. **Closed-form single-particle stencil.** Place one particle at a
     known position; hand-compute $\widetilde{\Delta}_n(\bm{x}_g;\Sigma)$
     at a few stencil grid points; assert `force_grid[i_x, i_y, i_z]`
     equals `F_n * Δ̃_n` to `sqrt(eps(T))`.
  3. **Stencil anchoring matches cuFCM convention.** For a particle at
     $Y = 0.3\Delta x$ with $M_G$ even, the stencil's lowest occupied
     grid index is $\mathrm{round}(Y/\Delta x) - M_G/2$; for
     $Y = 0.7\Delta x$ it is one index higher (after periodic wrap).
     Direct check of the nonzero-receiving indices.
  4. **Force conservation.** $\sum_g \widetilde{\mathcal{J}}^\dagger
     [\mathcal{F}](\bm{x}_g)\,\Delta x^3 \approx \sum_n \bm{F}_n$ within
     the truncation tolerance of the chosen $(M_G, \Sigma/\Delta x)$
     (paper Table 2, `outline.tex:600-612`).
  5. **First moment / centroid.** One unit particle:
     $\sum_g \bm{x}_g\, \widetilde{\Delta}_n(\bm{x}_g;\Sigma)\,\Delta x^3 \approx \bm{Y}_n$
     within the same truncation tolerance.
  6. **Periodicity.** Particle at $\bm{Y}_n$ vs $\bm{Y}_n + L\hat{e}_i$
     (after `wrap_positions!`) → identical `force_grid`.
  7. **Linearity in `F`.** $\widetilde{\mathcal{J}}^\dagger[\alpha\mathcal{F}_1 + \beta\mathcal{F}_2] = \alpha\widetilde{\mathcal{J}}^\dagger[\mathcal{F}_1] + \beta\widetilde{\mathcal{J}}^\dagger[\mathcal{F}_2]$.
  8. **Reflection symmetry.** Particle exactly on a grid point;
     `force_grid` is reflection-symmetric about that grid point along
     each axis, within `sqrt(eps(T))`.
  9. **Translation by one grid step.** Particle at $\bm{Y}_n$ vs
     $\bm{Y}_n + \Delta x\,\hat{e}_i$ → `force_grid` shifted by one cell
     on axis $i$.
  10. **Standard-FCM degenerate limit ($\Sigma = \sigma$).** With
      `Σ_over_σ = T(1)`, `spread_forces!` equals the closed-form
      standard-FCM Gaussian spread at every grid point.
- `test/test_spread_forces_inferred.jl` — `@inferred` type stability for
  `spread_forces!` and `_spread_forces_kernel!`, `Float32`/`Float64`.
- `test/test_spread_forces_allocations.jl` — `@ballocated == 0` for the
  wrapper and the kernel.
- `test/test_jet.jl` — `JET.@test_call spread_forces!` walks the full
  call graph for inference health.
- `test/test_aqua.jl` — `Aqua.test_all` covers the module hygiene.
- `test/test_cell_geometry.jl` (or `test/test_fcm_grid.jl`) pins the new
  cold-path derived fields.

Tolerances: the closed-form single-particle stencil, periodicity,
linearity, reflection, translation, and degenerate-limit tests use
`sqrt(eps(T))`. The truncation-tolerant tests (force conservation, first
moment) use the paper-tabulated error for the chosen $(M_G, \Sigma/\Delta x)$,
explicitly documented in the test body.
