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

### Step 3 kernel: force spreading

Force spreading ([force-spreading.md](force-spreading.md), the spreading operator of
§3 equation (25)) is ported to the device by adding a GPU method, dispatched on the
device buffer types, to the same `_spread_forces_kernel!` the CPU backend defines. The
public `spread_forces!` wrapper and `mobility!` stay backend-agnostic; the backend split
is the single dispatch seam on the buffer storage type.

The device kernel mirrors cuFCM's active block-per-particle shared-memory spreading
kernel, monopole (force) only:

- **One block per particle** (block-stride over particles for $N$ beyond the grid). Each
  block stages its particle's position and force in shared memory.
- **Separable precompute into shared memory.** The block's threads cooperatively fill, for
  the $M_G$ stencil points on each axis, the 1-D Gaussian weights
  $\texttt{inv\_norm} \cdot \exp(-x_i^2 \cdot \texttt{inv\_2Σ²})$, the axis-squared
  distances $x_i^2$, and the periodic-wrapped 1-based grid indices
  $\mathrm{mod}(g_i, M_i) + 1$. This block-local shared memory *replaces* the CPU's global
  `stencil` scratch — the device kernel does not read the `stencil` sub-struct buffers at
  all (the sub-struct table anticipates this: "may become block-local device memory").
- **Atomic scatter over the support patch.** The block's threads sweep the $M_G^3$ stencil
  points; each forms the modified-kernel weight $(a_0 + a_2 r^2)\, g_x g_y g_z$ and
  `atomic`-adds the force contribution to the three components of the grid force density.
  The atomic accumulation order is unspecified, so the spread grid matches the CPU's only
  to round-off (the parity tolerance) — the deterministic-reduction caveat the package
  already carries.

The polynomial coefficients $(a_0, a_2)$ and Gaussian normalisation come from the same
`_modified_kernel_coefficients` the CPU kernel uses (computed once on the host and passed
in), and the anchor $j_i = \mathrm{round}(Y_i \cdot \texttt{inv\_h})$ and the periodic
wrap follow the CPU conventions, so the two backends agree numerically. The force density
is zeroed per component before the launch, matching the CPU kernel's in-call zeroing. The
launch shape (threads per block, block cap) and the choice of shared memory over a
global-scratch or no-shared-memory variant are benchmark-gated; the initial port takes
cuFCM's block-per-particle shared-memory shape unchanged.

### Step 4 kernel: Stokes solve

The Fourier-space Stokes inversion ([stokes-solve.md](stokes-solve.md), the inverse Stokes
operator of §3 equations (32)–(33)) needs no backend-specific entry point — `stokes_solve!`
is already backend-agnostic. Its three forward and three inverse transforms go through the
`mul!(out, plan, in)` (AbstractFFTs) interface, which dispatches to the cuFFT R2C/C2R plans
the assembly builds on the device, so the transforms run on the GPU unchanged. The one
host-loop piece, the in-place per-mode projection `_apply_inverse_stokes_kernel!`, gets a
device method dispatched on the device buffer types — the single dispatch seam, mirroring
Step 3.

The whole step stays device-resident: the forward cuFFT writes the spectrum into `fluid_hat`
on the device, the projection kernel reads and writes it in place, and the inverse cuFFT
reads it straight back — nothing returns to the host.

The device kernel mirrors cuFCM's active `cufcm_flow_solve`, force (monopole) only:

- **One thread per Fourier mode** over the half-spectrum $(M_x/2+1) \times M_y \times M_z$, a
  grid-stride elementwise map — no shared memory, no atomics (the projection is purely local
  in the Fourier index, so the single in-place `fluid_hat` buffer is race-free). The launch
  takes cuFCM's 32 threads per block (`FCM_THREADS_PER_BLOCK`); the block count covers the
  spectrum.
- **Reused device wavenumbers.** Each thread reads $k_x[i_x], k_y[i_y], k_z[i_z]$ from the
  per-axis vectors the assembly copied to the device, exactly as the CPU kernel indexes its
  host vectors — the two backends see bit-identical wavevectors. (cuFCM recomputes the
  wavevector inline from the thread index because it never precomputes the vectors; reusing
  the device vectors is equivalent and keeps the tested wrap-layout in one place.)
- **Per-mode projection.** For $k^2 = \boldsymbol{k}\cdot\boldsymbol{k}$ the thread forms
  $\hat{\boldsymbol f} \leftarrow \frac{1}{\mu k^2 M}\,(\hat{\boldsymbol f} -
  \boldsymbol{k}\,(\boldsymbol{k}\cdot\hat{\boldsymbol f})/k^2)$ and writes the three
  components back, with the $\boldsymbol{k}=\boldsymbol 0$ mode a guarded write of zero (the
  mean-flow gauge fix). This is the same arithmetic as the CPU kernel; both cuFFT C2R and
  FFTW `brfft` are unnormalised, so the same $1/M$ round-trip factor applies.

The launch shape (32 versus a larger block, any change to the grid-stride mapping) is
benchmark-gated; the initial port takes cuFCM's shape unchanged.

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
- `test/cuda/test_gpu_spread_forces.jl` — CPU↔CUDA parity of the spread force density
  (compared to the documented tolerance, since the atomic scatter fixes the result but
  not the summation order) and `CUDA.@allocated == 0` for `spread_forces!`. The sorted
  position/force buffers are populated directly and identically on both backends, so the
  test isolates spreading from the cell-list sort.
- `test/cuda/test_gpu_stokes_solve.jl` — CPU↔CUDA parity of the Stokes-solved
  `fluid_velocity` (forward FFT → per-mode projection → inverse FFT, to the documented
  tolerance) and `CUDA.@allocated == 0` for `stokes_solve!`. The force density is written
  directly and identically on both backends, isolating the solve from spreading.

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

**Step 3 — spreading.** cuFCM's active spreading kernel
(`cufcm_mono_dipole_distribution_bpp_shared_dynamic`) distributes the monopole force and
the dipole/torque contributions in one block-per-particle pass, staging per-axis Gaussian
factors in shared memory and writing the grid with atomics. This package ports the same
block-per-particle shared-memory + atomic architecture but drops the dipole/torque terms
(force-only scope), staging and accumulating only the monopole force. The CPU-style
one-thread-per-particle spread (and any no-shared-memory variant) is a benchmark-gated
follow-up, not the initial port.

**Step 4 — Stokes solve.** cuFCM's `cufcm_flow_solve` recomputes each mode's wavevector
inline from the thread index and launches `FCM_THREADS_PER_BLOCK = 32` threads per block.
This package reuses the per-axis wavenumber vectors already resident on the device — the only
kernel-level divergence, keeping the CPU and GPU wavevectors bit-identical and the
wrap-layout derivation in one place — and keeps cuFCM's 32-thread launch. A larger block or
a different grid-stride mapping is a benchmark-gated follow-up.
