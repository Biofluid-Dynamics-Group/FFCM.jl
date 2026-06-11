# Particle Sorting and Cell Lists

Step 2 of the Fast FCM algorithm (Su & Keaveny 2024, §4).

## Summary

Given the spatial hashing of step 1, the arrays that hold the forces of the particles $\left\{ \boldsymbol{F}_n \right\}_{n = 1}^N$ and their positions $\left\{ \boldsymbol{Y}_n \right\}_{n = 1}^N$ are sorted by key so that they are stored in memory by cell.

In the code, we define
- `F` $= (\boldsymbol{F}_1, \dots, \boldsymbol{F}_N)$.

## Method

The particles are ordered by cell with a **counting sort** on the integer key
`cell_hash[n]`. Because the keys are bounded by the cell count, no comparison sort is
needed, and the per-cell index ranges fall out of the sort itself. The kernel proceeds in
three passes over the $N$ particles and the $m_x m_y m_z$ cells:

1. **Histogram.** Count how many particles fall in each cell.
2. **Exclusive prefix sum.** Turn the counts into the 1-based inclusive range
   $\text{cell\_start}[c] : \text{cell\_end}[c]$ of sorted slots that cell $c$ will occupy.
   An empty cell gets $\text{cell\_end}[c] = \text{cell\_start}[c] - 1$, an empty range.
3. **Stable scatter.** Walk the particles in ascending original index $n$ and place each
   into the next free slot of its cell, recording the permutation
   $\text{original\_index}[s] = n$ (sorted slot $s$ to original index $n$). Visiting $n$ in
   ascending order makes the sort **stable**: within a cell, particles keep their original
   relative order.

A final gather materialises the sorted data,
$$
\boldsymbol{Y}^{\text{sorted}}_s = \boldsymbol{Y}_{\text{original\_index}[s]},
\qquad
\boldsymbol{F}^{\text{sorted}}_s = \boldsymbol{F}_{\text{original\_index}[s]},
$$
so that particles sharing a cell occupy contiguous columns. That contiguity is the memory
locality the later steps rely on: the spread and interpolation (steps 3, 5) touch
overlapping grid patches for consecutive particles, and the pairwise correction (step 6)
walks a cell by its first and last slot.

## Contract

### Cold-path derived state (allocated once on `FFCMConfig`)

Sizes use `N` and the total cell count $m_x m_y m_z$.

- `original_index` (`Vector{Int32}`, length `N`) — the sort permutation, sorted slot to
  original particle index: $n = \text{original\_index}[s]$.
- `cell_start`, `cell_end` (`Vector{Int32}`, length $m_x m_y m_z$) — the per-cell
  **1-based inclusive** range into the sorted order. Cell $c \in 0{:}m_x m_y m_z - 1$
  occupies sorted slots `cell_start[c+1] : cell_end[c+1]`.
- `counting_sort_scratch` (`Vector{Int32}`, length $m_x m_y m_z$) — scratch for the counting sort
  (the histogram, then the scatter cursor). Not read by other steps.
- `Y_sorted`, `F_sorted` (`Matrix{T}`, shape `(3, N)`) — positions and forces gathered
  into the sorted order.

### Hot-path input (per `mobility!` call)

- `config` with a **current** `config.cell_hash`, i.e. `assign_cells!` has run since `Y`
  last changed (see [spatial-hashing.md](spatial-hashing.md)).
- `Y`, `F` (`AbstractMatrix{T}`, shape `(3, N)`) — particle positions and forces in the
  caller's original order.

### Hot-path output (written into `config`)

For every cell $c$, the slots $s \in \text{cell\_start}[c+1] : \text{cell\_end}[c+1]$
satisfy $\text{cell\_hash}[\text{original\_index}[s]] = c$; these ranges partition
$1{:}N$; and $\boldsymbol{Y}^{\text{sorted}}_s = \boldsymbol{Y}_{\text{original\_index}[s]}$
(likewise `F_sorted`). The sort is stable.

### Boundary case

An **empty cell** $c$ has $\text{cell\_end}[c+1] = \text{cell\_start}[c+1] - 1$, so its
range is empty. This holds for any occupancy — no separate fixup and no reliance on a
dense suspension.

## Implementation

`sort_particles_by_cell!(config, Y, F)` is the hot-path entry. Assuming
`config.cell_hash` is current, it builds the cell list and gathers the sorted data into
`config`, returning `config`. It is allocation-free and type-stable. It delegates to two
function-barrier kernels that take naked buffers, so each is independently testable and
fully specialised:

- `_build_cell_list_kernel!(original_index, cell_start, cell_end, counting_sort_scratch, cell_hash)`
  is the counting sort of Method: histogram into `counting_sort_scratch`, exclusive prefix sum into
  `cell_start`/`cell_end`, then the stable scatter into `original_index`.
- `_gather_particles_kernel!(Y_sorted, F_sorted, Y, F, original_index)` performs
  $\boldsymbol{Y}^{\text{sorted}}_s = \boldsymbol{Y}_{\text{original\_index}[s]}$ and the
  same for `F`.

The five cell-list buffers and the two sorted matrices are allocated once by the
`FFCMConfig` constructor.

| Phase | Allocations | Functions |
|---|---|---|
| Cold | OK | `FFCMConfig` constructor allocates `original_index`, `cell_start`, `cell_end`, `counting_sort_scratch`, `Y_sorted`, `F_sorted`. |
| Hot  | `@ballocated == 0` | `sort_particles_by_cell!(config, Y, F)` (after `wrap_positions!` + `assign_cells!`). |

The hot path is allocation-free and type-stable on `T <: AbstractFloat`.

## Performance notes

- The counting sort is $\mathcal{O}(N + m_x m_y m_z)$ and yields the cell ranges directly
  from its prefix sum, with no separate boundary scan.
- The histogram and scatter loops carry a dependency (random writes into `counting_sort_scratch` /
  `original_index`), so they are `@inbounds` but not `@simd`. The prefix-sum loop is a
  serial scan by construction.
- The bounds are safe: `cell_hash[n]` lies in $0{:}m_x m_y m_z - 1$ by step 1, and the
  scatter writes each slot $1{:}N$ exactly once.
- The gather reads `Y`/`F` by particle column (three contiguous components) and writes
  `Y_sorted`/`F_sorted` sequentially.
- `Int32` index width matches `cell_hash`; the total cell count fits well within `Int32`.
- **Caching opportunity (future).** The cell list depends only on positions (via
  `cell_hash`), while `F` changes every iteration of a downstream resistance solve. The
  `wrap → hash → build-cell-list` prefix could be cached across iterations with fixed
  `Y`, re-gathering only `F`. Not implemented now: `mobility!` rebuilds the list each call
  to keep a single public hot path.

## Verification

End-to-end correctness for this step is established by the test suite.

- `test/test_sort_particles_by_cell.jl` — `_build_cell_list_kernel!`: ascending-hash
  grouping, per-cell ranges bracket exactly their particles, empty-cell empty range,
  intra-cell stability; `_gather_particles_kernel!`: the gather equals
  `Y[:, original_index]`; and an end-to-end `wrap_positions!` → `assign_cells!` →
  `sort_particles_by_cell!` invariant on a random $N = 1000$ cloud for `Float32`/`Float64`.
- `test/test_cell_geometry.jl` — the `FFCMConfig` constructor allocates the cell-list
  buffers at the right sizes.
- `test/test_sort_particles_by_cell_inferred.jl` — `@inferred` type stability for the
  wrapper and both kernels, `Float32`/`Float64`.
- `test/test_sort_particles_by_cell_allocations.jl` — `@ballocated == 0` for the wrapper
  and both kernels.
- `test/test_jet.jl` — `JET.@test_call sort_particles_by_cell!` over the full call graph.
- `test/test_aqua.jl` — `Aqua.test_all` covers the module.

The sort and gather are exact integer and copy operations, so equality is exact; the
boundary cases (empty cells, stability) are pinned by explicit tests rather than a
tolerance.

## Differences from cuFCM

> Comparison against the C++/CUDA reference implementation, kept for validation during
> development and removed once the port is complete.

The cuFCM reference (`cuFCM/src/CUFCM_CELLLIST.cu`: `sort_index_by_key`,
`create_cell_list`) was audited against this design.

| Facet | cuFCM | This package |
|---|---|---|
| Sort algorithm | `cub::DeviceRadixSort::SortPairs` (keys = hash, values = particle index) | counting sort (histogram → exclusive prefix → stable scatter) |
| Build `cell_start`/`cell_end` | a separate kernel scanning sorted hashes for boundaries | a byproduct of the prefix sum; no second pass |
| Index base / end semantics | 0-based; `cell_end` exclusive | 1-based; `cell_end` inclusive |
| Empty cells | left undefined (relies on dense packing) | well-defined empty range `cell_end = cell_start − 1` |
| Permutation direction | `index`: sorted → original | `original_index`: same |
| Temp storage | `key_buf`, `index_buf`, temp storage malloc'd per call | pre-allocated `counting_sort_scratch` on `config`; allocation-free |
| Stability | radix sort is stable | scatter in ascending $n$ is stable |
| Data reorder | gathers position/force (and more) via `index` | gathers `Y`/`F` into `Y_sorted`/`F_sorted` |
| Neighbour-cell map | `bulkmap_loop` precomputes neighbour indices | out of scope here — that is step 6 (pairwise correction) |

The one behavioural difference worth flagging is empty-cell handling: cuFCM's
boundary scan writes `cell_start`/`cell_end` only for occupied cells, leaving empty cells
undefined (harmless in the dense suspensions it targets). The counting sort here writes
every cell, so an empty cell is a well-defined empty range and the pairwise correction can
iterate any cell without a guard.
