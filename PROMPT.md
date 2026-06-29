# Handoff — validate C2 (GPU cell list) on a GPU node

You are continuing work on **FFCM.jl** (branch `cuda`). The previous session
implemented **C2 — GPU pipeline step 1 (spatial hashing + cell list + neighbour
map)** *off-host*, on a machine with no GPU. **Nothing GPU has been run.** Your job
is to run the test suite on this GPU node, fix anything that the off-host build got
wrong, and bring C2 to green.

Read first: `PLAN.md` (C2 entry + decisions), `spec/cuda-conventions.md`,
`spec/particle-sorting.md`, `spec/spatial-hashing.md`. This is TDD: the failing test
`test/cuda/test_gpu_cell_list.jl` drove the implementation.

## What was built

- `src/config.jl` — `CellBuffers{IntVector, NeighborMap}` gained a second type param:
  `neighbor_map` is `nothing` on the CPU and a device vector on the GPU. The
  constructor builds the host neighbour map (`_build_neighbor_map`) in the GPU branch.
- `src/cell_list.jl` — `_build_neighbor_map(num_cells)` host helper (13 half-shell
  neighbours per cell, integer `mod`). CPU step functions unchanged.
- `ext/FFCMCUDAExt/` — the extension is now a **directory** (`FFCMCUDAExt.jl` +
  `config.jl` + `cell_list.jl`). `cell_list.jl` holds the device methods, dispatched on
  the device buffer types, of `wrap_positions!`, `_assign_cells_kernel!`,
  `_build_cell_list_kernel!` (parallel **atomic counting sort**: histogram →
  `accumulate!` prefix sum → atomic-cursor scatter), and `_gather_particles_kernel!`.
- `test/cuda/test_gpu_cell_list.jl` — CPU↔CUDA parity (wrap, hashes, ranges,
  permutation partition invariants, gathered data, neighbour map) + an allocation check.

## Run

This node has 8 shared GPUs and no queue. **Pin one free card first** — the test
harness skips every GPU test unless exactly one device is visible (it will not grab a
card someone else is using). Pick a free GPU and export it before running:

```bash
gpustat                         # or nvidia-smi — find a free card
export CUDA_VISIBLE_DEVICES=<index>
julia --project=test -e 'using Pkg; Pkg.test()'
```

The suite prints the card it selected (name + free memory) before the GPU tests; if it
reports "N GPUs visible and none pinned", the export above was missed. The CPU suite
must stay green (the `CellBuffers` type-param change + the directory move touch shared
code). The new `test/cuda/test_gpu_cell_list.jl` fires once a single device is pinned;
the existing `test/cuda/test_gpu_construction.jl` now also asserts
`config.cells.neighbor_map isa CuArray`.

To iterate faster on just the GPU tests:

```bash
julia --project=test -e 'using FFCM, CUDA; include("test/test_utilities.jl"); include("test/cuda/test_gpu_cell_list.jl")'
```

## Known risks — fix-forward

1. **First real `@cuda` JIT.** C1 flagged a possible toolkit/driver
   mismatch (`ERROR_NOT_SUPPORTED`) under memory pressure in some machines. This session is where the
   first kernels compile. If JIT fails, resolve the toolchain (align the CUDA runtime
   CUDA.jl uses with the driver; `CUDA.versioninfo()`), not the kernels.
2. **`accumulate!` allocates for large cell counts.** Verified from CUDA.jl source:
   the prefix sum is allocation-free only on the single-block scan path; a large cell
   count takes the multi-block scan, which mallocs a transient aggregate buffer. The
   allocation test deliberately uses a small `N`/cell count (single-block, `@allocated
   == 0`). If it nonetheless allocates, that is the documented gap — a custom
   allocation-free scan is the follow-up (do NOT block C2 on it; note it).
3. **Atomic API spelling.** Scatter uses
   `CUDA.atomic_add!(pointer(cursor, i), Int32(1))` (returns the old value = the slot)
   and histogram uses `CUDA.@atomic counts[i] += Int32(1)`. Both verified present in
   CUDA.jl 6.2; if a signature drifted, adjust to the installed version.
4. **Parity exactness.** Cell-list parity uses in-box positions so the wrap is the
   exact identity and hashes compare bit-for-bit; the wrap parity test (out-of-box)
   uses `isapprox` (CPU/GPU `mod` may differ by a ULP). If a hash mismatch appears for
   in-box input, suspect a `floor(Int32, ·)` / clamp divergence in
   `_assign_cells_device!`.
5. **Permutation order.** The atomic scatter is non-stable, so `original_index` is only
   checked by partition invariants (permutation of `1:N`, `cell_hash[perm]` sorted), not
   exact equality with the CPU. `cell_start`/`cell_end` *are* compared exactly (they
   depend only on per-cell counts).

## On green

- Tick **C2** in `PLAN.md` (change `[~]` to `[x]`, drop the "pending validation" note;
  keep the backlog items). Record the run outcome in `CHANGELOG.md` (dev journal).
- Run `/compact`, then `/clear` after updating `PLAN.md` (per `CLAUDE.local.md`).
- Delete this `PROMPT.md` (it is a session handoff, not a shipped artefact).
- Next session is **C3 — GPU force spreading**.
