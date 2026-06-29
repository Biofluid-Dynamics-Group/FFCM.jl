# CUDA Backend Conventions

The conventions governing the GPU backend of the Fast FCM mobility operator
(Su & Keaveny 2024, §3 and §4). This document is the cross-cutting reference for
the CUDA backend: backend selection, the buffer/plan ownership boundaries, and
the CPU↔CUDA parity contract. Per-step kernel conventions (shared-memory tiling,
atomics policy, launch shapes) are recorded here as each pipeline step's GPU
kernel is validated; until then this document specifies only the backend
scaffolding that is in place.

## Summary

The mobility operator $\mathcal{M}^{\mathcal{V}\mathcal{F}}$ (§3, equation (19))
is backend-agnostic in its mathematics: the CPU and GPU backends compute the
same six-step splitting ([mobility.md](mobility.md)) to round-off. They may
differ *architecturally* — a GPU is a throughput machine, so the device kernels
are organised for parallel occupancy rather than transcribed from the CPU loops
— but they must agree *numerically*. CPU↔CUDA parity tests bridge the two
implementations; the published paper (§3, §4) remains the specification for the
algorithm both backends compute.

The backend is selected once, at construction, by the `gpu_acceleration` keyword
of `FFCMConfig`. The choice is encoded in the configuration's storage types, so
it never reaches the `mobility!` hot path.

## Method

### Backend selection is a construction-time type choice

`FFCMConfig` owns every per-call buffer. The buffers must live where the kernels
run: host memory for the CPU backend, device memory for the GPU backend. The
backend is therefore fixed when the buffers are allocated — at construction —
and recorded in the configuration's *type* through the storage-type parameters
of its sub-structs ([mobility.md](mobility.md), "Backend selection"). Because the
type carries the backend, the step functions dispatch on their buffer arguments
with no runtime branch: a config holding device arrays selects the device
kernels, a config holding host arrays selects the host kernels, and `mobility!`
itself is identical source for both.

The user expresses this as a single intent flag, `gpu_acceleration::Bool`, rather
than naming a device array type. The flag hides the CUDA vocabulary from callers
who do not have CUDA loaded, and routes — cold path only — to the per-backend
buffer assembly. It is consumed during construction and is not a field of the
configuration.

### Sub-struct boundaries

The compiled buffers are grouped by *kind of information*, not by pipeline phase,
into five storage-type-parameterized sub-structs:

| Sub-struct | Holds | Backend-specific growth |
|---|---|---|
| `cells` | cell-list integer bookkeeping (hash, sort permutation, per-cell ranges, counting-sort workspace) | the neighbour map — a second type parameter, `nothing` on the CPU and the device half-shell map on the GPU |
| `particles` | the $3 \times N$ wrapped/sorted position and force buffers | — |
| `grid` | the real-space force-density and fluid-velocity fields | — |
| `solver` | the Fourier-space field, the wavevectors, and the FFT plans | cuFFT plans replace FFTW plans |
| `stencil` | per-particle stencil workspace | may become block-local device memory |

Grouping by information keeps each boundary a coherent unit a single backend
owns, and lets a backend carry fields the other does not need (for example, the
device neighbour map) without the CPU configuration paying for them.

### The assembly seam

The constructor runs three phases: validate the parameters, derive the
backend-independent scalars and lookup tables, then assemble the backend buffers
and plans. Only the third phase is backend-specific. The CPU assembly lives in
`src`; the GPU assembly is supplied by the package extension. The constructor
routes to one or the other on the `gpu_acceleration` flag.

### Step 1 kernels: hashing, counting sort, neighbour map

The first pipeline step (spatial hashing and the cell list,
[spatial-hashing.md](spatial-hashing.md), [particle-sorting.md](particle-sorting.md))
is ported to the device by adding GPU methods, dispatched on the device buffer
types, to the same step functions the CPU backend uses (`wrap_positions!`,
`_assign_cells_kernel!`, `_build_cell_list_kernel!`, `_gather_particles_kernel!`).
The public wrappers (`assign_cells!`, `sort_particles_by_cell!`) and `mobility!`
stay backend-agnostic; the backend split is a single dispatch seam on the buffer
storage type.

- **Wrap and hash** are grid-stride kernels, one thread per particle: fold each
  position into $[0, L_i)$, then compute the same floor / clamp / linearisation as
  the CPU hash. The per-axis $\min(\cdot, m_i - 1)$ clamp is kept.
- **Cell list** is a GPU **counting sort** — the same algorithm as the CPU
  ([particle-sorting.md](particle-sorting.md)), parallelised: an atomic-increment
  histogram into the counting-sort workspace, an exclusive prefix sum to the
  per-cell ranges, and an atomic-cursor scatter of the permutation. The per-cell
  ranges depend only on the per-cell counts, so `cell_start`/`cell_end` are
  identical to the CPU's; the scatter is **not** stable (atomic ordering), so the
  intra-cell order of `original_index` — and hence the column order of the gathered
  `Y_sorted`/`F_sorted` — is unspecified and may vary run to run. This is
  numerically immaterial (the downstream sums over a cell are order-independent to
  round-off) and matches the deterministic-reduction caveat the package already
  carries. Counting sort needs exactly the buffers the CPU backend already owns, so
  it adds no device-only cell-list field.
- **Neighbour map** (the half-shell of 13 forward neighbours per cell, consumed by
  the pairwise correction) is geometry-only, so it is built once on the host by the
  cold-path helper `_build_neighbor_map` and copied to the device. It is the single
  backend-divergent `cells` field: a second type parameter, `nothing` on the CPU
  (the CPU correction computes neighbours on the fly) and a device vector on the GPU.

The prefix sum uses CUDA.jl's `accumulate!`, which is allocation-free for a
single-block scan but allocates a transient aggregate buffer for the multi-block
scan of a large cell count; an allocation-free scan for large grids is a deferred
follow-up (recorded below).

## Contract

### `gpu_acceleration` keyword

- `gpu_acceleration::Bool = false` on `FFCMConfig(; …)` and `FFCMConfig{T}(; …)`.
- `false` builds a CPU-backed configuration with no dependency on CUDA.
- `true` builds a GPU-backed configuration. It requires the CUDA backend to be
  available — the `CUDA` package loaded (`using CUDA`, which loads the extension)
  on a machine with a functional CUDA device. Without the extension loaded,
  construction throws `ArgumentError` naming the requirement.
- `gpu_acceleration = true` with `T === Float64` emits a non-fatal warning:
  double-precision throughput is a fraction of single-precision on consumer
  NVIDIA hardware, so `Float32` is the intended GPU precision, but `Float64` is
  permitted.

### CPU↔CUDA parity

The GPU backend is correct when, for the same positions and forces, it reproduces
the CPU backend's velocities to the documented tolerance ($\sqrt{\texttt{eps}(T)}$
relative, with the absolute floor for near-zero entries). Parity is asserted per
pipeline step as each step's GPU kernel lands, and end-to-end for the assembled
operator. Parity tests run only where `CUDA.functional()` is true and are skipped
on a host without a CUDA device.

### GPU extension entry point

The extension supplies the GPU buffer/plan assembly by adding a method to the
`_assemble_gpu_buffers` function declared in `src`. The method allocates the
device-backed sub-structs and the cuFFT plans, and returns them in the same order
the CPU assembly does. The `src`-side declaration carries only a fallback that
throws when the extension is not loaded, so `src` never names a CUDA type.

## Implementation

CUDA appears only in `ext/FFCMCUDAExt.jl`, never in `src` (CLAUDE.md §7). The
extension is loaded automatically when a user runs `using CUDA` alongside `using
FFCM`, through the `[weakdeps]`/`[extensions]` tables of the package.

The backend routing avoids a runtime branch on the hot path by construction. The
cold-path constructor may branch and may be type-unstable in its return type
(the returned configuration is concretely typed either way); only the hot path is
held to the static-dispatch and allocation-free contracts. When CUDA is not
loaded, the `_assemble_gpu_buffers` fallback is the only method, so it always
throws and the `gpu_acceleration = true` branch infers to `Union{}` — the
constructor remains fully inferred on the default CPU path.

GPU construction launches no Julia kernel: it allocates the device buffers and
creates the cuFFT plans, neither of which JIT-compiles a kernel. The buffers are
left unzeroed (the CPU backend zeros only because FFTW's measuring planner dirties
its buffers; cuFFT planning does not), since every buffer is written before it is
read on a `mobility!` call. This keeps construction independent of the device's
kernel-JIT path, which matters on GPUs the toolkit only partially supports.

The device kernels are hand-written `@cuda` kernels (CLAUDE.md §4). The GPU hot
path is held to `CUDA.@allocated == 0`, mirroring the CPU `@ballocated == 0`
contract, once the kernels are in place.

## Performance notes

- `Float32` is the intended GPU precision; `Float64` runs but is slow on consumer
  cards (the `cc 6.1` local validation target has FP64 at a small fraction of FP32
  throughput).
- `fft_threads` is a CPU-only concept (it bakes a thread count into the FFTW
  plans); the GPU backend ignores it.
- Per-step kernel performance choices (shared-memory tiling, atomics, launch
  shapes, register pressure) are benchmark-driven and recorded in this document
  as they are validated against the faithful baseline, not assumed up front.

## Verification

- `test/cuda/` — guarded by `CUDA.functional()`, skipped on a GPU-less host:
  a GPU-backed `FFCMConfig` constructs with device-backed sub-struct buffers and
  cuFFT plans, and `Float64 + gpu_acceleration = true` emits the warning.
- The not-loaded error — `gpu_acceleration = true` without the CUDA extension
  throws `ArgumentError` — is checked without a GPU, in a process that has not
  loaded `CUDA`.
- CPU↔CUDA parity fixtures in `test/test_utilities.jl` build a GPU configuration
  mirroring the standard CPU test configuration and compare results; they are
  exercised per step as the GPU kernels land.

## Differences from cuFCM

> Comparison against the C++/CUDA reference implementation, kept for validation
> during development and removed once the port is complete.

The GPU backend faithfully mirrors the architecture of the active (non-commented)
kernels in the cuFCM reference (`cuFCM/src/`): a single device-resident pipeline —
device sort and cell-start/end build, shared-memory block-per-particle spreading
and gathering with atomics and block reductions, the cuFFT Stokes solve, and a
pair-once real-space correction over a precomputed neighbour map. The only
admitted divergence is scope: this package implements the force→velocity coupling
$\mathcal{M}^{\mathcal{V}\mathcal{F}}$ only, dropping the dipole/torque couplings,
the rotational degrees of freedom, and the random-displacement machinery that
cuFCM also carries. Performance-motivated deviations from the reference
architecture are deferred and benchmark-gated, not baked into the initial port;
validated outcomes are recorded in this document.

Unlike the CPU backend, which differs architecturally from cuFCM (loops, not
block-per-particle kernels), the GPU backend tracks the reference closely; the
two FFCM.jl backends are reconciled numerically by the parity tests, not
structurally.

**Step 1 — sort algorithm.** cuFCM sorts particles by cell key with
`cub::DeviceRadixSort` (O(N), stable). CUDA.jl exposes no radix sort, and its
stdlib `sortperm!` is a bitonic network (O(N log²N), ≈ 100× the radix work at
N = 10⁶). This package instead ports the CPU **counting sort** to the device
(O(N + cells)): same algorithmic result (particles grouped by cell, ranges from a
prefix sum), O(N), and no device-only buffers. The trade is a non-stable atomic
scatter, which makes the intra-cell order nondeterministic (immaterial to the
velocities, per the deterministic-reduction caveat). A stable, deterministic
O(N) device **radix** sort — matching cuFCM exactly — is a benchmark-gated
follow-up, worth revisiting if reproducibility is required or atomic contention
bottlenecks at extreme cell occupancy. cuFCM also precomputes the neighbour map
(`bulkmap`); computing the half-shell on the fly in the GPU correction would drop
the only backend-divergent field — also a benchmark-gated follow-up.
