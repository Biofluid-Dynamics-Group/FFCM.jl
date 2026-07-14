# GPU acceleration

The CUDA backend runs the whole pipeline — cell list, spreading, FFT Stokes solve,
interpolation, and pairwise correction — resident on an NVIDIA GPU. It lives in a
package extension that loads automatically when CUDA.jl is present, so the package has
no GPU dependency unless you ask for one.

## Usage

Load CUDA alongside FFCM and pass `gpu_acceleration = true` at construction:

```julia
using CUDA
using FFCM

config = FFCMConfig(;
    L = (250.0f0, 250.0f0, 250.0f0),
    num_grid_points = Int32.((256, 256, 256)),
    kernel_widths_ratio = 2.0f0,
    M_G = 10,
    R_c = 5.6419f0,
    N = 186_510,
    viscosity = 1.0f0,
    gpu_acceleration = true,
)

V = zeros(Float32, 3, N)
mobility!(V, config, Y, F)   # Y, F, V are ordinary host arrays
```

The backend is chosen once, at construction, and encoded in the configuration's storage
types — it is not stored as a flag, and [`mobility!`](@ref) contains no backend branch.
Everything else about the API is unchanged: the same [`FFCMConfig`](@ref) keywords, the
same [`mobility!`](@ref) call, and the same [`FFCMMobility`](@ref) operator, which
drives a GPU configuration through `mul!` exactly as it drives a CPU one.

## What to expect

- **Host arrays in, host arrays out.** `Y`, `F`, and `V` are ordinary ``3 \times N``
  host matrices for both backends. A GPU configuration uploads the inputs and downloads
  the velocities internally; you never allocate or handle a device array.
- **CUDA must be loaded and functional.** `gpu_acceleration = true` without `using CUDA`
  (or without a working CUDA device) throws an `ArgumentError` naming the requirement.
- **`Float32` is the intended GPU precision.** `Float64` is permitted but emits a
  warning at construction: consumer NVIDIA cards run double precision at a small
  fraction of single-precision throughput.
- **CPU↔GPU agreement.** For the same positions and forces, the GPU backend reproduces
  the CPU backend's velocities to the documented round-off tolerance
  (``\sqrt{\mathrm{eps}(T)}`` relative). The two backends differ only in floating-point
  summation order (GPU kernels accumulate atomically or in parallel), never in the
  algorithm. The test suite asserts this parity per pipeline step and end to end
  whenever a CUDA device is available.
- **The hot path allocates nothing on the device**, mirroring the CPU allocation-free
  guarantee.
- **`fft_threads` is a CPU concept** (it configures the FFTW plans) and is ignored by
  the GPU backend, whose transforms run through cuFFT.

How the device kernels are organised — launch shapes, shared-memory staging, the
host↔device boundary — is documented in the
[GPU architecture](devdocs/gpu-architecture.md) page of the developer documentation.
