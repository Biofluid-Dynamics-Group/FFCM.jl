# Spatial Hashing

Step 1 of the Fast FCM algorithm (Su & Keaveny 2024, §4).

## Summary

The domain for the Stokes problem is $\Omega = [0, L_x) \times [0, L_y) \times [0, L_z)$
with triply-periodic boundary conditions. Fast FCM is accelerated by decoupling the
discretisation level of the spectral solver from the width of the delta distribution
approximation.

The grid for the spectral solver is a uniform discretisation with $M_x, M_y, M_z$ gripoints in
directions $z, y, z$, with spacings $h_x, h_y, h_z$. The total number of gridpoints is then $M = M_x M_y M_z$.

The first step in acceleration involves a grouping of the $N$ particles at positions $\left\{ \boldsymbol{Y}_n \right\}_{n = 1}^N$ in the domain into
_cells_: a partition of the domain into rectangular prisms. The cells are generated so that
all particles within a _cutoff radius_ $R_c$ of a specific particle are either in the same
cell or in adjacent cells. Axis $x$ is divided into $m_x$ intervals, axis $y$ into $m_y$ and axis $z$ into $m_z$: that determine the total amount of cells. Each cell has an index where
$$
  \text{cell index} = i + \left(j + k m_y\right) m_x \quad \text{(§4, equation (68))}
$$
with $i = 0, \dots, m_x - 1$, $j = 0, \dots m_y - 1$ and $k = 0, \dots, m_z - 1$. Each particle is assigned to its cell (by position) identified with the cell index. The hashing is then a mapping
particle $\mapsto$ cell index.

In the code, we define
- `L` $= (L_x, L_y, L_z)$,
- `N` $= N$,
- `Y` $= (\boldsymbol{Y}_1, \dots, \boldsymbol{Y}_N)$,
- `R_\c` $= R_c$, 
- `num_cells` $= (m_x, m_y, m_z)$ and
- `cell_hash[n]` $= \text{cell index of particle } n$.

## Contract

### Cold-path input (user-supplied to `FFCMConfig`)

- `L::NTuple{3, T}` $= (L_x, L_y, L_z)$. Corner-origin matches the
  paper convention and the cuFCM C++ reference, so every step can be
  cross-checked against the paper without a coordinate translation in the
  way.
- Cutoff radius `R_c::T` $= R_c$ for the pairwise correction.
- Particle count `N::Integer` $= N$. Fixed for the lifetime of the
  `FFCMConfig`; changing `N` requires a new configuration.

### Cold-path derived state (computed once, stored on `FFCMConfig`)

- `num_cells::NTuple{3, Int32}` — number of cells per axis, with
  components `num_cells[i]` $= m_i =$ `max(floor(Int32, L_i / R_c), Int32(3))`.
  The `max(·, 3)` floor guarantees that for any particle position the
  26-neighbour stencil covers all `R_c`-balls (paper §4 Step 1).
- `cell_size::NTuple{3, T}` — cell extents `cell_size[i]` $= L_i / m_i$.
  Invariant `cell_size_i ≥ R_c` by construction.
- `inv_cell_size::NTuple{3, T}` — precomputed `1 / cell_size[i]` so the hot
  path multiplies instead of divides.
- `cell_hash::Vector{Int32}` — length-`N` buffer that the hot path writes
  into `cell_hash[n]` is the cell index of particle `n`.

### Hot-path input (per `mobility!` call)

- `Y::AbstractMatrix{T}` of shape `(3, N)` — particle positions. Throughout
  this spec, paper notation is used: `i ∈ {x, y, z}` (equivalently
  $i ∈ \{1, 2, 3\}$) is the Cartesian axis index, and `n ∈ {1, …, N}` is
  the particle index. The Julia layout maps to $Y^n_i = $ `Y[i, n]`.
  Positions outside $\Omega$ are tolerated and folded by `wrap_positions!`.

### Hot-path output

- `cell_hash[n] = x_c + (y_c + z_c · m_y) · m_x` for `n ∈ 1:N`. The encoding
  lays `x` out fastest and `z` slowest, matching
  `cuFCM/src/CUFCM_CELLLIST.cu:create_hash_gpu`.

### Periodicity contract

`wrap_positions!` is the canonical wrap point and is the first call inside
`mobility!`. After it returns, every downstream step may assume
$Y^n_i \in [0, L_i)$ for every particle `n` and every axis `i`. The hash
kernel additionally clamps each cell coordinate to `m_i - 1` to defend
against the fp-roundoff corner case where `mod(y, L_i)` rounds to
numerically `L_i`.

Boundary cases (post-wrap, for any particle `n`):

- $Y^n_i = 0$ (lower closed edge) → cell coordinate $0$ on axis $i$.
- $Y^n_i = L_i$ (open upper edge, reachable only by fp roundoff) → cell
  coordinate $m_i - 1$ after clamp.

## API

```julia
"""
    FFCMConfig{T}(; L, R_c, N)

Cold-path configuration of the Fast FCM mobility operator. Owns the cell
geometry derived from the periodic domain `L = (L_x, L_y, L_z)` and the
cutoff `R_c`, plus the hot-path buffers sized for `N` particles. The
configuration is built once and reused across many `mobility!` calls; all
per-call work writes into pre-allocated buffers owned by this struct.
"""
struct FFCMConfig{T <: AbstractFloat}
    L::NTuple{3, T}
    R_c::T
    num_cells::NTuple{3, Int32}
    cell_size::NTuple{3, T}
    inv_cell_size::NTuple{3, T}
    cell_hash::Vector{Int32}
end

"""
    wrap_positions!(Y, L) -> Y

Fold each column of `Y` (a `3×N` matrix of particle positions) into the
canonical periodic domain `[0, L_a)` for each axis `a`. Idempotent. The
fp-roundoff corner case where `mod(y, L)` rounds to exactly `L` is left
to the downstream cell-index clamp in `_assign_cells_kernel!`.
"""
function wrap_positions!(Y::AbstractMatrix{T}, L::NTuple{3, T}) where {T} end

"""
    assign_cells!(cfg, Y) -> cfg.cell_hash

Write each particle's cell index into `cfg.cell_hash`. Assumes `Y` has
already been folded into the canonical domain by `wrap_positions!`. The
hash linearises the 3-D cell coordinate with `x` fastest and `z` slowest.
"""
function assign_cells!(cfg::FFCMConfig{T}, Y::AbstractMatrix{T}) where {T} end
```

The underscore-prefixed
`_assign_cells_kernel!(cell_hash, Y, inv_cell_size, num_cells)` is the
function-barrier kernel: it takes naked tuples and buffers, so it is
independently testable and benefits from Julia's standard type-stability
pattern.

## Cold-path vs hot-path

| Phase | Allocations | Functions |
|---|---|---|
| Cold | OK | `FFCMConfig` constructor: derive `num_cells`, `cell_size`, `inv_cell_size`; allocate `cell_hash`. |
| Hot  | `@ballocated == 0` | `wrap_positions!(Y, L)` then `assign_cells!(cfg, Y)`. |

The hot path is allocation-free and type-stable on `T <: AbstractFloat`.

## Performance notes

- `Y::AbstractMatrix{T}` of shape `(3, N)` interacts well with Julia's
  column-major layout: per particle `n`, the three reads `Y[1, n]`,
  `Y[2, n]`, `Y[3, n]` are contiguous.
- The hash kernel avoids division by precomputing `inv_cell_size`. The
  `min` clamp lowers to `vminss`/`vminsd` on x86 — branch-free and
  SIMD-compatible.
- Both passes use `@inbounds @simd for n in axes(Y, 2)`. Bounds are safe
  because `cell_hash` was allocated to length `N = size(Y, 2)` at
  construction and `Y` is iterated by its own column axis.
- `Int32` is the cell-index width; `num_cells ≤ 2^31` covers any realistic
  FCM run (matches cuFCM's `int` width).

## Diffs from cuFCM

The cuFCM reference (`cuFCM/src/CUFCM_CELLLIST.cu`,
`CUFCM_SOLVER.cu`, `CUFCM_DATA.cu`) was audited against this design.

| Facet | cuFCM | This package |
|---|---|---|
| Domain origin | `[0, L_a)` | same |
| Hash linearisation | `x_c + (y_c + z_c·m_y)·m_x` | same |
| Cell-count derivation | `max(L_i/R_c, 3)` then cast to int | `max(floor(Int32, L_i/R_c), 3)` (equivalent for positive arguments) |
| Anisotropy | supported via grid dims | supported via `L::NTuple{3, T}` |
| Hash index type | `int` (32-bit) | `Int32` |
| Memory layout | AoS `Y[3*np + k]` | column-major `Matrix{T}(3, N)` (identical access pattern) |
| Position wrap | separate `box<<<>>>` kernel mutates `Y` in place | `wrap_positions!` mutates `Y` in place (same model) |
| Upper-edge roundoff | `if(x == boxsize) x = 0` inside `images()` | cell-index clamp `min(·, num_cells - 1)` inside the kernel |
| Cell-count safety floor | `max(·, 3)` in C++ | same |
| Function-pointer indirection (`linear_encode`/`icell`) | yes | dropped — overkill for our use |

The single behavioural difference is the location of the upper-edge
roundoff fix: cuFCM patches the wrapped position; we clamp the cell
index. Both produce a valid hash in the corner case; the cell-index clamp
keeps `wrap_positions!` itself branch-light.

## Verification

End-to-end correctness for this step is established by the test suite:

- `test/test_cell_geometry.jl` — `FFCMConfig` constructor: derived
  `num_cells`, `cell_size`, `inv_cell_size`, the tiny-box `max(·, 3)`
  floor, and argument validation.
- `test/test_wrap_positions.jl` — `wrap_positions!`: identity on
  already-wrapped input, periodicity, idempotence, edge cases including
  the upper-boundary `Y = L` → `0` parity with cuFCM's `images()` fixup.
- `test/test_assign_cells_kernel.jl` — `_assign_cells_kernel!`:
  hand-computed hashes for known cell layouts; anisotropic stride;
  upper-edge roundoff clamp regression.
- `test/test_assign_cells.jl` — `assign_cells!` outer wrapper agrees
  with the kernel.
- `test/test_assign_cells_inferred.jl` — `@inferred` type stability
  for `Float32` and `Float64`.
- `test/test_assign_cells_allocations.jl` — `@ballocated == 0` for
  both passes and the kernel.
- `test/test_aqua.jl` — `Aqua.test_all` against the module (method
  ambiguity, stale [deps]/[extras], project consistency).
- `test/test_jet.jl` — `JET.@test_call` on each hot-path entry point
  for `Float32` and `Float64`, walking the full call graph.

Tolerances: spatial hashing is integer arithmetic; cell-index equality is
exact. Float-roundoff cases are pinned by explicit boundary tests.
