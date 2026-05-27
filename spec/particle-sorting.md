# Particle Sorting and Cell Lists

Step 2 of the Fast FCM algorithm (Su & Keaveny 2024, §4, Step `sort`;
`paper/tex/outline.tex:518`). Given the per-particle cell hash from step 1,
this step (a) counting-sorts the particles by hash so that particles sharing
a cell — and particles spreading to common grid points — become contiguous
in memory, and (b) records, for each cell, the index range of its particles
in the sorted order. The contiguity makes spreading/interpolation (steps 3,
5) cache-friendly and lets the pairwise correction (step 6) walk a cell by
its first/last index.

## Contract

### Cold-path derived state (allocated once on `FFCMConfig`)

Sizes use `N` (particle count) and `num_cells_total = m_x · m_y · m_z`.

- `original_index::Vector{Int32}` (length `N`) — the sort permutation,
  sorted-slot → original-particle-index: `n = original_index[s]`.
- `cell_start::Vector{Int32}`, `cell_end::Vector{Int32}` (length
  `num_cells_total`) — per-cell **1-based inclusive** index range into the
  sorted order. Cell `c ∈ 0:num_cells_total-1` occupies sorted slots
  `cell_start[c+1] : cell_end[c+1]`.
- `cell_cursor::Vector{Int32}` (length `num_cells_total`) — scratch for the
  counting sort (histogram, then scatter cursor). Not read by other steps.
- `Y_sorted::Matrix{T}`, `F_sorted::Matrix{T}` (shape `(3, N)`) — positions
  and forces gathered into the sorted order.

### Hot-path input (per `mobility!` call)

- `config` with a **current** `config.cell_hash`, i.e. `assign_cells!` has
  run since `Y` last changed (see [spatial-hashing.md](spatial-hashing.md)).
- `Y::AbstractMatrix{T}`, `F::AbstractMatrix{T}` of shape `(3, N)` —
  particle positions and forces in the caller's (original) order. Paper
  notation: `i ∈ {x, y, z}` axis, `n ∈ {1, …, N}` particle, `Y[i, n]`.

### Hot-path output (written into `config`)

For every cell `c ∈ 0:num_cells_total-1`, the slots
`s ∈ cell_start[c+1]:cell_end[c+1]` satisfy
`cell_hash[original_index[s]] == c`, these ranges partition `1:N`, and
`Y_sorted[:, s] == Y[:, original_index[s]]` (likewise `F_sorted`). The sort
is **stable**: within a cell, `original_index` lists particles in ascending
original index.

Boundary case: an **empty cell** `c` has `cell_end[c+1] = cell_start[c+1]-1`,
so `cell_start[c+1]:cell_end[c+1]` is an empty range. This holds for any
occupancy — no separate fixup and no reliance on a dense suspension.

## API

```julia
"""
    sort_particles_by_cell!(config, Y, F) -> config

Counting-sort the particles by cell hash and gather their positions and
forces into sorted order. Assumes `config.cell_hash` is current. On return
`config.original_index`, `config.cell_start`/`cell_end`,
`config.Y_sorted`/`F_sorted` hold the cell list and sorted data.
"""
function sort_particles_by_cell!(config::FFCMConfig{T},
                                 Y::AbstractMatrix{T},
                                 F::AbstractMatrix{T}) where {T} end
```

Two underscore-prefixed function-barrier kernels take naked buffers so they
are independently testable and fully specialized:

- `_build_cell_list_kernel!(original_index, cell_start, cell_end,
  cell_cursor, cell_hash)` — the counting sort: histogram cell occupancy
  into `cell_cursor`, exclusive-prefix-sum into `cell_start`/`cell_end`,
  then a stable scatter of particle indices into `original_index`.
- `_gather_particles_kernel!(Y_sorted, F_sorted, Y, F, original_index)` —
  `Y_sorted[:, s] = Y[:, original_index[s]]` and likewise for `F`.

## Cold-path vs hot-path

| Phase | Allocations | Functions |
|---|---|---|
| Cold | OK | `FFCMConfig` constructor allocates `original_index`, `cell_start`, `cell_end`, `cell_cursor`, `Y_sorted`, `F_sorted`. |
| Hot  | `@ballocated == 0` | `sort_particles_by_cell!(config, Y, F)` (after `wrap_positions!` + `assign_cells!`). |

The hot path is allocation-free and type-stable on `T <: AbstractFloat`.

## Performance notes

- **Counting sort** is `O(N + num_cells_total)` and yields the cell ranges
  directly from its prefix sum — no separate boundary scan. The integer keys
  are bounded by `num_cells_total`, so no comparison sort is needed.
- The histogram and scatter loops carry a dependency (random writes into
  `cell_cursor` / `original_index`), so they are `@inbounds` but not
  `@simd`. The prefix-sum loop is a serial scan by construction.
- Bounds are safe: `cell_hash[n] ∈ 0:num_cells_total-1` by step 1, and the
  scatter writes each slot `1:N` exactly once, so `@inbounds` holds.
- The gather walks `Y`/`F` by particle column (contiguous reads of three
  components) and writes `Y_sorted`/`F_sorted` sequentially.
- `Int32` index width matches `cell_hash` and cuFCM's `int`;
  `num_cells_total ≤ 2^31` covers any realistic run.
- **Caching opportunity (future):** the cell list depends only on positions
  (via `cell_hash`), while `F` changes every iteration of the downstream
  resistance solve. When `mobility!` exists, the
  `wrap → hash → build-cell-list` prefix can be cached across iterations
  with fixed `Y`, re-gathering only `F`. Not implemented now: there is no
  `mobility!` to host the cache, and building speculative infrastructure
  would violate the YAGNI discipline in `CLAUDE.md`.

## Diffs from cuFCM

The cuFCM reference (`cuFCM/src/CUFCM_CELLLIST.cu`: `sort_index_by_key`,
`create_cell_list`) was audited against this design.

| Facet | cuFCM | This package |
|---|---|---|
| Sort algorithm | `cub::DeviceRadixSort::SortPairs` (keys = hash, values = particle index) | counting sort (histogram → exclusive prefix → stable scatter) |
| Build `cell_start`/`cell_end` | separate `create_cell_list` kernel scanning sorted hashes for boundaries | byproduct of the prefix sum; no second pass |
| Index base / end semantics | 0-based; `cell_end` exclusive (`== N` or next start) | 1-based; `cell_end` inclusive last |
| Empty cells | entries left undefined (relies on dense packing) | empty range `cell_end = cell_start - 1`, correct for any occupancy |
| Permutation direction | `index`: sorted → original | `original_index`: same |
| Temp storage | `key_buf`, `index_buf`, `d_temp_storage` malloc'd per call | pre-allocated `cell_cursor` on `config`; allocation-free |
| Stability | radix sort is stable | scatter in ascending `n` is stable |
| Data reorder | gathers position/force (and more) via `index` | gathers `Y`/`F` into `Y_sorted`/`F_sorted` |
| 13-half-neighbour cell map | `bulkmap_loop` precomputes neighbour indices | out of scope — that is step 6 (pairwise correction) |

The one behavioural difference worth flagging is empty-cell handling:
cuFCM's boundary-scan only writes `cell_start`/`cell_end` for occupied
cells, leaving empty cells undefined (harmless in the dense suspensions it
targets). Our counting sort writes every cell, so an empty cell is a
well-defined empty range — the pairwise correction can iterate any cell
without a guard.

## Verification

End-to-end correctness for this step is established by the test suite:

- `test/test_sort_particles_by_cell.jl` — `_build_cell_list_kernel!`:
  ascending-hash grouping, per-cell ranges bracket exactly their particles,
  empty-cell empty range, intra-cell stability; `_gather_particles_kernel!`:
  gather equals `Y[:, original_index]`; and an end-to-end
  `wrap_positions!` → `assign_cells!` → `sort_particles_by_cell!` invariant
  on a random `N = 1000` cloud for `Float32`/`Float64`.
- `test/test_cell_geometry.jl` — `FFCMConfig` allocates the cell-list
  buffers at the right sizes.
- `test/test_sort_particles_by_cell_inferred.jl` — `@inferred` type
  stability for the wrapper and both kernels, `Float32`/`Float64`.
- `test/test_sort_particles_by_cell_allocations.jl` — `@ballocated == 0`
  for the wrapper and both kernels.
- `test/test_jet.jl` — `JET.@test_call sort_particles_by_cell!` walks the
  full call graph for inference health.
- `test/test_aqua.jl` — `Aqua.test_all` (package hygiene) covers the module.

Tolerances: the sort and gather are exact integer/copy operations; equality
is exact. Boundary cases (empty cells, stability) are pinned by explicit
tests rather than tolerances.
