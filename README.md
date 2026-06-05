# FFCM.jl

Julia implementation of the Fast Force-Coupling Method (FFCM) ([Su & Keaveny (2024), *Accelerating the
force-coupling method for hydrodynamic interactions in periodic
domains*, J. Comput. Phys. **510**, 113060](https://www.sciencedirect.com/science/article/pii/S0021999124003097)), for
hydrodynamic interactions between rigid spheres in a triply-periodic
Stokes flow.

FFCM is a numerical approximation to the mobility tensor of a collection of identical spherical rigid particles
suspended in a fluid that obeys the Stokes equation with periodic boundary conditions. Given $N$ spherical particles of radius $a=1$ located at $\left(\boldsymbol{Y}_n\right)_{n = 1}^N \subset \Omega$
with forces $\left(\boldsymbol{F}_n\right)_{n = 1}^N$ acting on them, FFCM computes
their resulting velocities $\left(\boldsymbol{V}_n\right)_{n = 1}^N$ considering hydrodynamic interactions.

## Usage

The package provides three objects for plug-and-play usage of the method. `FFCMConfig` provides
the setup for the problem, where the physics (domain size, number of particles, viscosity) and
numerical parameters (spectral method grid, local stencil size, method parameters) can be set.

`FFCMMobility` is the actual mobility operator, and `mobility!` is a mutating function that acts on its first argument, populating it with the
velocities given by the arguments (both interfaces share the same backend).

```julia
using FFCM

T = Float32  # or Float64

L_x = T(10)  # Set \Omega = [0, 10)^3
L_y = T(10)
L_z = T(10)

μ = T(1)  # Viscosity

R_c = T(1)  # Cutoff radius
Σ_over_σ = T(2)  # FFCM splitting parameter

Y = T[9.3 4.1 -0.4 4.0; 4.0 4.2 4.0 9.9; 4.0 4.0 4.0 4.0]  # Positions 3xN
F = T[0.5 -0.3 0.2 0.1; -0.1 0.4 0.0 0.3; 0.2 0.1 0.7 -0.5]  # Forces 3xN
N = Y.shape[2]  # Number of particles

num_points = Int32(16)  # Grid discretisation
M_G = 8  # Stencil size

config = FFCMConfig{T}(
    L = (L_x, L_y, L_z),
    R_c = R_c,
    N = N,
    Σ_over_σ = Σ_over_σ,
    μ = μ,
    num_grid_points = (num_points, num_points, num_points),
    M_G = M_G
)

V = zeros(T, 3, N)  # Target array for velocities, size 3xN of type T
mobility!(V, config, Y, F)  # Use FFCM as a function

M = FFCMMobility(config, Y)  # Mobility operator
V = M*F  # Alternatively, use the method as a linear operator (useful for iterative solvers)
```

## Parameter optimisation

## GPU acceleration

## License

MIT — see [LICENSE](LICENSE).

## LLM assistance

This repository was written with assistance of `claude-4.7-opus` and `claude-4.6-sonnet`. [CLAUDE.md](CLAUDE.md) serves as agentic guidance for this project.