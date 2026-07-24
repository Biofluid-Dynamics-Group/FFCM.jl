# ForceCouplingMethod.jl benchmark suite

The regression detector for the FFCM mobility pipeline: no performance-related
change ships without a before/after measurement from this suite. Timings are
hardware-specific, so results live in the git-ignored `results/` directory and
comparisons are made between two runs on the same machine.

## Running

From the repository root:

```sh
julia -t auto --project=benchmark benchmark/run.jl
```

This runs the full suite (a few minutes plus one-time compilation) and writes
`benchmark/results/<short-sha>.json` together with a `.info` sidecar recording
the machine, Julia version, and thread count. Pass a path to choose the output
file. For a quick laptop run, restrict the problem sizes:

```sh
FFCM_BENCHMARK_SIZES=small julia -t auto --project=benchmark benchmark/run.jl
```

## Comparing two runs

Record a baseline at the reference commit, make the change, run again, then:

```sh
julia --project=benchmark benchmark/judge.jl results/<after>.json results/<baseline>.json
```

Each leaf's minimum time is classified as a regression, an improvement, or
invariant under BenchmarkTools' default 5% time tolerance. Compare only runs
from the same machine, Julia version, and thread count (check the `.info`
sidecars).

**Thermal state matters.** On laptops, minutes of sustained benchmarking
throttle the CPU well below its boost clock, which shifts *every* leaf by
tens of percent between runs; a uniform shift across steps a change cannot
have touched is the signature. Judge only runs taken in a similar thermal
state, and prefer in-process side-by-side comparisons (like the
`fft-planning` group) when comparing code variants — interleaved
measurements in one process see the same clocks.

## What is measured

The suite is a `BenchmarkGroup` tree keyed as `backend / group / precision /
size`, with `Float32` and `Float64` leaves at every size:

| Group | Measures |
|---|---|
| `cpu/construction` | `FFCMConfig` construction (cold path, including FFT planning at the default effort), one evaluation |
| `cpu/mobility` | the assembled `mobility!` call — the per-iteration cost of a downstream resistance solve, and the headline number |
| `cpu/steps/<step>` | each of the seven pipeline steps in isolation, keyed by entry-point name, to localise which cost centre a change moved |
| `cpu/fft-threads` | `stokes_solve!` with serial plans vs plans threaded across `julia -t`'s thread count |
| `cpu/fft-planning` | opt-in (see below): `stokes_solve!` under `:estimate` vs `:measure` plans, side by side |

Problem sizes hold the particle volume fraction at ≈ 5% (semi-dilute,
unit-radius particles) while the grid doubles per step, so the FFT and the
spread/interpolate costs grow together with physically consistent loading:

| Size | Box `L` | Grid | Particles `N` |
|---|---|---|---|
| `small` | 16³ | 32³ | 50 |
| `medium` | 32³ | 64³ | 400 |
| `large` | 64³ | 128³ | 3200 |

Positions and forces are seeded (`Xoshiro`), so two runs of the same code
measure identical work.

## The planner-effort experiment

`FFCM_BENCHMARK_FFT_PLANNING=true` adds the `cpu/fft-planning` group, which
times the FFT-based Stokes solve under `:estimate` and `:measure` plans on
the same problems. Run it **separately** from regression runs: FFTW consults
its in-process wisdom at every planning level, so once a measured plan has
been built, every later plan of the same transform in the same process —
including the `construction` leaves — silently benefits from it. For the same
reason, the one-time cost of measured planning is only honest in a fresh
process; wisdom can be carried across sessions deliberately with
`FFTW.import_wisdom`/`FFTW.export_wisdom` around `FFCMConfig` construction.

The same effect deflates the `construction` leaves whenever the default
planner effort is a measuring level: the suite has already built
identically-shaped plans by the time those leaves run, so they price a
warm-wisdom construction. The honest first-construction cost of a session is
a fresh-process measurement, recorded in the development changelog.

## Hardware flexibility

Nothing in the suite hardcodes a machine: sizes are selected with
`FFCM_BENCHMARK_SIZES`, threading follows `julia -t`, and judge comparisons
are relative to a baseline recorded on the same hardware rather than to
absolute thresholds.

## Reference-implementation baseline

[`cufcm-baseline.md`](cufcm-baseline.md) records a one-time measurement of
cuFCM — the paper authors' reference CUDA implementation — on an RTX 2080 Ti,
together with the exact configuration and its ForceCouplingMethod.jl equivalent. Future
`cuda/` leaves at that configuration can be compared against it directly.
