# cuFCM reference baseline (RTX 2080 Ti, 2026-07-13)

A one-time timing record of **cuFCM** — the paper authors' reference CUDA
implementation (<https://github.com/racksa/cuFCM>; Su & Keaveny 2024, *J. Comput.
Phys.* 510, 113060, §5–6) — taken on the same card and configuration used to
cross-validate FFCM.jl's GPU backend. It preserves the setup and the numbers so
future `cuda/` benchmark leaves can be compared against the reference without
keeping a cuFCM checkout around.

At the time of measurement, FFCM.jl reproduced cuFCM's velocities on identical
inputs to round-off (Float64 mean per-particle relative error 3.1e-15, Float32
3.1e-6), so these timings are a like-for-like anchor for the configuration below,
not merely a similar workload.

## Hardware and build

- GeForce RTX 2080 Ti (sm_75, 11 GB), NVIDIA driver 460.27.04, host `nvidia4`.
- cuFCM built with nvcc 11.2.67, `-arch=sm_75 -std=c++14 -O3` (**note:** the
  upstream makefile default is `-O0`; timings at `-O0` are not comparable), g++
  9.3 host compiler. Single precision = the plain `CUFCM` target; double = the
  `-DUSE_DOUBLE_PRECISION` build.

## Configuration

Physical parameters (paper §5 symbols), the ε ≈ 1e-4 accuracy regime of paper
Tables 1–2 at the Table-3 semi-dilute operating point:

| quantity | value |
|---|---|
| box `L` (cubic, periodic) | 250 |
| grid `M` (cubic) | 256³ (`Δx = L/M = 0.9765625`) |
| particle radius `a` | 1 (`σ = a/√π ≈ 0.5641896`) |
| `Σ/σ` | 2 |
| `M_G` | 10 |
| `R_c` | 5.641895835477563 (= 10 σ, i.e. λ_ε = R_c/Σ = 5) |
| viscosity `μ` | 1 (hardcoded in cuFCM) |
| `N` | 186,510 (volume fraction φ = 5 %) |

Inputs: positions uniform in `[0, L)³`, force components uniform in
`[-0.5, 0.5]`, torques zero (translational problem), drawn from
`Xoshiro(20260713)`. At this `N` the timings are insensitive to the seed.

cuFCM configures the same physics through three dimensionless knobs — its
config-file values were derived from the parameters above and are recorded here
for exact reproduction:

| cuFCM knob | meaning | value |
|---|---|---|
| `alpha` | Σ/Δx | 1.155460267105805 |
| `beta` | M_G/alpha | 8.654559818874588 |
| `eta` | R_c/Σ | 5.0 |
| `nx = ny = nz` | grid | 256 |
| `boxsize` | L | 250 |
| `rh` | a | 1 |

The equivalent FFCM.jl construction (what a future `cuda/` leaf should time):

```julia
config = FFCMConfig(;
    L = (250.0f0, 250.0f0, 250.0f0),
    num_grid_points = Int32.((256, 256, 256)),
    kernel_widths_ratio = 2.0f0,
    M_G = 10,
    R_c = 5.6418959f0,
    N = 186_510,
    viscosity = 1.0f0,
    gpu_acceleration = true,
)
```

## cuFCM timings

Per-application means over post-warmup repeats, from cuFCM's own timers (each
region bracketed by `cudaDeviceSynchronize`; the first 20 % of repeats are
warmup). Float32: 50 repeats; Float64: 5 repeats.

| step | Float32 (s) | Float64 (s) |
|---|---|---|
| hashing | 0.000944 | 0.001114 |
| spreading | 0.004721 | 0.017274 |
| FFT | 0.006721 | 0.029799 |
| gathering | 0.003263 | 0.018192 |
| correction | 0.006911 | 0.026179 |
| **compute** | **0.021616** | **0.091445** |

Measurement semantics, needed for honest comparison:

- `compute` = spreading + FFT + gathering + correction. It **excludes hashing
  and all host↔device transfers** — cuFCM uploads positions/forces once, outside
  its repeat loop.
- cuFCM's headline metric is PTPS = N / compute: **8.63e6** (Float32),
  2.04e6 (Float64 — Turing runs FP64 at 1/32 rate).
- The like-for-like target for FFCM.jl's `mobility!` (which re-hashes and stages
  Y/F/V on every call) is **hashing + compute = 0.022561 s** (Float32), i.e.
  PTPS 8.27e6.

## FFCM.jl standing at measurement time (context)

Measured the same day, same card, same inputs (Julia 1.12.6, CUDA.jl 5.8.5,
CUDA runtime 11.8): `mobility!` Float32 wall time min 0.020196 s / mean
0.020624 s → PTPS 9.24e6, **111.7 % of the cuFCM hashing+compute throughput**,
with per-call staging included and before the deferred perf work (upload
caching, device-array fast path, allocation-free large-grid scan). Float64:
min 0.072509 s.

## cuFFT library-vintage experiment (2026-07-14)

Attribution experiment for the throughput gap above: is FFCM.jl's edge explained
by its newer cuFFT (10.9.0.58, from the CUDA.jl 11.8 runtime artifact) versus the
toolkit-11.2 cuFFT (10.4.0.72) the cuFCM binary links? Both ship the soname
`libcufft.so.10`, and `ldd` shows it is the binary's **only** dynamic CUDA
library (cudart is static), so an `LD_LIBRARY_PATH` override swaps exactly one
variable:

```
libcufft.so.10 => /usr/local/cuda/targets/x86_64-linux/lib/libcufft.so.10   (A: baseline)
libcufft.so.10 => <repo>/cuFCM/compare/cufft-11.8/libcufft.so.10            (B: artifact 10.9)
```

Same card, config, and seeded inputs as the baseline; four runs per precision
interleaved A/B/A/B; per-step means over the binary's post-warmup repeats
(Float32 ×50, Float64 ×5). Mean-B/mean-A ratios:

| step | Float32 | Float64 |
|---|---|---|
| hashing (control) | 0.990 | 0.942 |
| spreading (control) | 0.999 | 0.931 |
| **FFT** | **1.015** | 0.957 |
| gathering (control) | 0.997 | 0.948 |
| correction (control) | 0.997 | 0.910 |
| compute | 1.003 | 0.937 |

**Outcome: the library-vintage hypothesis is refuted.** On Float32 — the
precision where the 111.7 % headline was measured, with control steps tight at
≤ 1 % — cuFFT 10.9 makes the binary's FFT **1.5 % slower**, not faster. The
Float64 legs moved −4…−9 % *including every non-FFT control step*, which cannot
be a cuFFT effect; at 5 repeats per run that is run-to-run/clock noise, not
signal. Velocity drift under the swap is indistinguishable from the atomic
nondeterminism floor (Float32: mean 1.4e-7 vs floor 1.5e-7; Float64: 3.9e-16 vs
3.6e-16), so the two cuFFT versions are numerically interchangeable here.

The remaining attribution hypotheses for FFCM.jl's kernel-level edge are, in
order: codegen vintage (LLVM 18 + ptxas 11.8 versus nvcc 11.2 across every
kernel) and monopole-specialized kernels (no runtime `rotation` argument or
dipole register/shared-memory footprint).

A correct benchmark will also go through 50 repeats in Float32 to warmup the card, and
this gates an actual performance test against cuFCM. These numbers are only initial.