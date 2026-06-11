# Spatial Hashing

Step 1 of the Fast FCM algorithm (Su & Keaveny 2024, §4).

## Summary

The domain for the Stokes problem is $\Omega = [0, L_x) \times [0, L_y) \times [0, L_z)$
with triply-periodic boundary conditions. Fast FCM is accelerated by decoupling the
discretisation level of the spectral solver from the width of the delta distribution
approximation.

The grid for the spectral solver is a uniform discretisation with $M_x, M_y, M_z$ grid points in
directions $x, y, z$, with uniform spacing $h$ (equal across axes). The total number of grid points is then $M = M_x M_y M_z$.

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
- `R_c` $= R_c$,
- `num_cells` $= (m_x, m_y, m_z)$ and
- `cell_hash[n]` $=$ the cell index of particle $n$.

## Method

### Cell geometry and the covering guarantee

The number of cells per axis is
$$
m_i = \max\!\left(\left\lfloor \frac{L_i}{R_c} \right\rfloor, 3\right)
\qquad (\text{paper §4, Step 1}).
$$
This lets the pairwise correction (step 6) find every neighbour within $R_c$ by
inspecting only a particle's own cell and its 26 immediate neighbours — the
$3 \times 3 \times 3$ block centred on it. Why the block suffices splits into two
regimes, depending on which term of the $\max$ is active.

**The box is at least three cutoffs wide on axis $i$** ($L_i/R_c \geq 3$). Then
$m_i = \lfloor L_i/R_c \rfloor \leq L_i/R_c$, so each cell is at least as wide as the
cutoff, $L_i/m_i \geq R_c$. Take a particle at $\boldsymbol{Y}_n$ in the cell with
coordinates $(c_x, c_y, c_z)$; any particle within $R_c$ of it differs by at most
$R_c \leq L_i/m_i$ — one cell width — on each axis, so its cell coordinate on axis $i$
differs from $c_i$ by at most one. The $3 \times 3 \times 3$ block therefore contains
the whole $R_c$-ball.

**The box is narrower than three cutoffs on axis $i$** ($2 \leq L_i/R_c < 3$, the range
still permitted by the pairwise-correction precondition $R_c \leq \min(L)/2$). The floor
clamps to $m_i = 3$, and the cell may now be narrower than the cutoff, $L_i/m_i < R_c$.
Coverage holds for a different reason: with only three cells on the axis, the neighbour
offsets $\{-1, 0, +1\} \bmod 3 = \{0, 1, 2\}$ span every cell, so the
$3 \times 3 \times 3$ block sweeps the whole axis and no neighbour can be missed.

In both regimes the floor $m_i \geq 3$ also guarantees that, under periodicity, the 26
neighbours are distinct cells, so none is double-counted as its own neighbour (with
$m_i = 2$ the offsets $\{-1, +1\} \bmod 2$ would collide).

### Hashing a position to a cell

A wrapped position $\boldsymbol{Y}_n \in [0, L_i)$ maps to the cell coordinate
$$
c_i = \min\left(\left\lfloor Y_{n,i} \frac{1}{\text{cell size}_i} \right\rfloor, m_i - 1\right),
$$
and the cell index linearises the three coordinates with $x$ fastest and $z$ slowest, as
in the Summary,
$$
\text{cell\_hash}[n] = c_x + (c_y + c_z m_y) m_x .
$$
The $\min(\cdot, m_i - 1)$ clamp guards one floating-point corner: if $Y_{n,i}$ is just
below $L_i$ and $Y_{n,i}/\text{cell size}_i$ rounds up to exactly $m_i$, the floor
would give $m_i$ — one past the last cell — and the clamp folds it back to $m_i - 1$.

### Periodic wrapping

Positions are folded into $[0, L_i)$ by $Y_{n,i} \mapsto \operatorname{mod}(Y_{n,i}, L_i)$
before hashing. This is the canonical wrap point for the whole pipeline: after it, every
later step may assume $\boldsymbol{Y}_n \in \Omega$.

## Contract

### Cold-path input (user-supplied to `FFCMConfig`)

- `L` $= (L_x, L_y, L_z)$ (`NTuple{3, T}`) — the periodic box. The corner origin matches
  the paper convention, so every step can be cross-checked against the paper without a
  coordinate shift.
- `R_c` $= R_c$ (`T`) — the cutoff radius for the pairwise correction.
- `N` $= N$ (`Integer`) — the particle count, fixed for the lifetime of the
  `FFCMConfig`; changing it requires a new configuration.

### Cold-path derived state (computed once, stored on `FFCMConfig`)

- `num_cells` $= (m_x, m_y, m_z)$ (`NTuple{3, Int32}`), with
  `num_cells[i] = max(floor(Int32, L_i / R_c), Int32(3))`.
- `cell_size` $= (L_i / m_i)$ (`NTuple{3, T}`) — the cell extents, $\geq R_c$ when
  $L_i/R_c \geq 3$ and $L_i/3$ otherwise (see the covering guarantee in Method).
- `inv_cell_size` $= (1 / \text{cell size}_i)$ (`NTuple{3, T}`) — precomputed so the hot
  path multiplies instead of divides.
- `cell_hash` (`Vector{Int32}`, length `N`) — the buffer the hot path writes each
  particle's cell index into.

### Hot-path input (per `mobility!` call)

- `Y` (`AbstractMatrix{T}`, shape `(3, N)`) — the particle positions, `Y[i, n]`.
  Positions outside $\Omega$ are tolerated and folded by `wrap_positions!`.

### Hot-path output

- `cell_hash[n]` $= c_x + (c_y + c_z m_y) m_x$ for $n \in 1{:}N$, with the cell
  coordinates of Method. The encoding lays $x$ out fastest and $z$ slowest.

### Periodicity contract

`wrap_positions!` is the canonical wrap point and the first call inside `mobility!`.
After it returns, every downstream step may assume $\boldsymbol{Y}_n \in [0, L_i)$. The
hash additionally clamps each cell coordinate to $m_i - 1$ to defend against the
fp-roundoff corner where $\operatorname{mod}(Y, L_i)$ rounds up to numerically $L_i$.

### Boundary cases (post-wrap, for any particle $n$)

- $Y_{n,i} = 0$ (lower closed edge) → cell coordinate $0$ on axis $i$.
- $Y_{n,i} = L_i$ (open upper edge, reachable only by fp roundoff) → cell coordinate
  $m_i - 1$ after the clamp.

## Implementation

`assign_cells!(config, Y)` is the hot-path entry for this step: it writes each
particle's cell index into `config.cell_hash`, assuming `Y` has already been folded into
$\Omega$. It delegates to the function-barrier kernel
`_assign_cells_kernel!(cell_hash, Y, inv_cell_size, num_cells)`, which takes naked buffers
and tuples so it is type-stable and independently testable; the kernel performs the floor,
the clamp, and the linearisation of Method.

Positions are folded by `wrap_positions!`, which has two forms. `wrap_positions!(dest, src, L)`
writes the wrapped positions of `src` into a separate buffer `dest`, so `mobility!` can
wrap into a `config`-owned scratch without touching the caller's array; the in-place
`wrap_positions!(Y, L)` folds `Y` itself. Both are idempotent, and the fp-roundoff corner
where $\operatorname{mod}(Y, L)$ rounds to exactly $L$ is left to the cell-index clamp
rather than handled here.

The cell geometry (`num_cells`, `cell_size`, `inv_cell_size`) and the `cell_hash` buffer
are built once by the `FFCMConfig` constructor.

| Phase | Allocations | Functions |
|---|---|---|
| Cold | OK | `FFCMConfig` constructor: derive `num_cells`, `cell_size`, `inv_cell_size`; allocate `cell_hash`. |
| Hot  | `@ballocated == 0` | `wrap_positions!(Y, L)` then `assign_cells!(config, Y)`. |

The hot path is allocation-free and type-stable on `T <: AbstractFloat`.

## Performance notes

- Storing `Y` as a `(3, N)` matrix suits Julia's column-major layout: per particle the
  three reads `Y[1, n]`, `Y[2, n]`, `Y[3, n]` are contiguous.
- The hash avoids division by multiplying by the precomputed `inv_cell_size`. The `min`
  clamp lowers to a branch-free `vminss`/`vminsd` on x86, so it stays SIMD-compatible.
- Both passes use `@inbounds @simd for n in axes(Y, 2)`. The bounds are safe because
  `cell_hash` was allocated to length `N = size(Y, 2)` and `Y` is iterated by its own
  column axis.
- `Int32` is the cell-index width; the total cell count fits well within `Int32` for any
  realistic FCM run.

## Verification

End-to-end correctness for this step is established by the test suite.

- `test/test_cell_geometry.jl` — the `FFCMConfig` constructor: the derived `num_cells`,
  `cell_size`, `inv_cell_size`, the small-box $\max(\cdot, 3)$ floor, and argument
  validation.
- `test/test_wrap_positions.jl` — `wrap_positions!`: identity on already-wrapped input,
  periodicity, idempotence, and the edge cases, including the upper-boundary
  $Y = L \to 0$ behaviour.
- `test/test_assign_cells_kernel.jl` — `_assign_cells_kernel!`: hand-computed hashes for
  known cell layouts, anisotropic stride, and the upper-edge roundoff clamp.
- `test/test_assign_cells.jl` — the `assign_cells!` wrapper agrees with the kernel.
- `test/test_assign_cells_inferred.jl` — `@inferred` type stability for `Float32` and
  `Float64`.
- `test/test_assign_cells_allocations.jl` — `@ballocated == 0` for both passes and the
  kernel.
- `test/test_aqua.jl` — `Aqua.test_all` (method ambiguity, stale deps, project
  consistency).
- `test/test_jet.jl` — `JET.@test_call` on each hot-path entry point for `Float32` and
  `Float64`, walking the full call graph.

Spatial hashing is integer arithmetic, so cell-index equality is exact; the
floating-point edge cases are pinned by explicit boundary tests rather than a tolerance.

## Differences from cuFCM

> Comparison against the C++/CUDA reference implementation, kept for validation during
> development and removed once the port is complete.

The cuFCM reference (`cuFCM/src/CUFCM_CELLLIST.cu`, `CUFCM_SOLVER.cu`, `CUFCM_DATA.cu`) was
audited against this design.

| Facet | cuFCM | This package |
|---|---|---|
| Domain origin | $[0, L_i)$ | same |
| Hash linearisation | $c_x + (c_y + c_z m_y) m_x$ | same |
| Cell-count derivation | `max(L_i/R_c, 3)` then cast to int | `max(floor(Int32, L_i/R_c), 3)` (equivalent for positive arguments) |
| Anisotropy | supported via grid dims | supported via `L::NTuple{3, T}` |
| Hash index type | `int` (32-bit) | `Int32` |
| Memory layout | AoS `Y[3⋅np + k]` | column-major `Matrix{T}(3, N)` (identical access pattern) |
| Position wrap | a separate kernel mutates `Y` in place | `wrap_positions!` (in-place form mutates `Y`; out-of-place form writes a scratch buffer) |
| Upper-edge roundoff | `if (x == boxsize) x = 0` inside `images()` | cell-index clamp `min(⋅, m_i − 1)` inside the kernel |
| Function-pointer indirection (`linear_encode`/`icell`) | yes | dropped — unnecessary here |

The single behavioural difference is the location of the upper-edge roundoff fix: cuFCM
patches the wrapped position, while this package clamps the cell index. Both produce a
valid hash in the corner case; the cell-index clamp keeps `wrap_positions!` branch-light.
