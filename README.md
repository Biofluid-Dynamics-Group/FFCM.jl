# FFCM.jl

Julia implementation of the Fast Force-Coupling Method (FFCM) for
hydrodynamic interactions between rigid spheres in a triply-periodic
Stokes flow.

The implementation follows [Su & Keaveny (2024), *Accelerating the
force-coupling method for hydrodynamic interactions in periodic
domains*, J. Comput. Phys. **510**, 113060](https://www.sciencedirect.com/science/article/pii/S0021999124003097).

## Intended interface

A CUDA backend loads automatically when [CUDA.jl](https://github.com/JuliaGPU/CUDA.jl)
is available in the user's environment (via a package extension; no CUDA
dependency for CPU users):

## Documentation map

- [CLAUDE.md](CLAUDE.md) — agent operating guide for this repo.
- [spec/](spec/) — algorithmic specifications
- [test/](test/) — unit, accuracy (paper-derived), API, and CUDA tests.
- [bench/](bench/) — performance benchmarks.

## Reference implementation

A CUDA/C++ reference, [`cuFCM`](https://github.com/racksa/cuFCM),
is used for performance hints (memory layout, cell-list, FFT
orchestration). Its naming and code style differ from the paper; this
package follows the paper.

## License

MIT — see [LICENSE](LICENSE).

## LLM assistance

This repository was written with assistance of `claude-4.7-opus` and `claude-4.6-sonnet`. [CLAUDE.md](CLAUDE.md) serves as agentic guidance for this project.