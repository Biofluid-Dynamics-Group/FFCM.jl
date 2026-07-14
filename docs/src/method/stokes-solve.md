# The spectral Stokes solve

Step 4 of the pipeline (Su & Keaveny 2024, §4) solves the periodic Stokes problem driven
by the spread force density ``\boldsymbol{f} = \tilde{\mathcal{J}}^\dagger[\mathcal{F}]``,

```math
-\mu \Delta \boldsymbol{u} + \nabla p = \boldsymbol{f},
\qquad
\operatorname{div}(\boldsymbol{u}) = 0
\qquad \text{in } \Omega
\qquad (\text{§3, equations (23)–(24)}),
```

with triply-periodic boundary conditions, by a Fourier spectral method: transform the
force density, invert the Stokes operator mode by mode, and transform back.

## Fourier-space inversion

The Fourier transform turns the differential operators into multiplications. Writing
``\hat{\boldsymbol{u}}``, ``\hat{p}``, ``\hat{\boldsymbol{f}}`` for the transforms and
``k = \lvert\boldsymbol{k}\rvert``,

```math
\mu k^2 \hat{\boldsymbol{u}} + i \boldsymbol{k} \hat{p} = \hat{\boldsymbol{f}},
\qquad
i \boldsymbol{k} \cdot \hat{\boldsymbol{u}} = 0 .
```

Dotting the momentum equation with ``\boldsymbol{k}`` and using incompressibility
eliminates the velocity and gives the pressure,
``\hat{p} = -i\, (\boldsymbol{k} \cdot \hat{\boldsymbol{f}})/k^2``. Substituting back
yields the projector form

```math
\hat{\boldsymbol{u}}(\boldsymbol{k})
= \frac{1}{\mu k^2}
\left(\boldsymbol{I} - \frac{\boldsymbol{k}\otimes\boldsymbol{k}}{k^2}\right)
\hat{\boldsymbol{f}}(\boldsymbol{k}),
\qquad \boldsymbol{k} \neq \boldsymbol{0}:
```

each mode is projected onto the divergence-free subspace and scaled by ``1/(\mu k^2)``.

## The mean-flow gauge fix

At ``\boldsymbol{k} = \boldsymbol{0}`` the momentum equation reduces to
``\boldsymbol{0} = \hat{\boldsymbol{f}}(\boldsymbol{0})``: the solvability condition is
that the total integrated force vanishes, and when it does, the mean velocity is
undetermined — the solution is fixed only up to a uniform translation. The standard
gauge choice sets ``\hat{\boldsymbol{u}}(\boldsymbol{0}) = \boldsymbol{0}``, the frame
in which the volume-averaged fluid velocity vanishes. For a physical mobility problem
with ``\sum_n \boldsymbol{F}_n = \boldsymbol{0}`` the discarded mean is zero up to the
stencil truncation of the spread; any residual mean is absorbed by the gauge fix and
does not affect any non-zero mode.

## Discrete transforms and normalisation

The implementation uses real-to-complex (r2c) forward and complex-to-real (c2r) backward
transforms, whose composition is unnormalised: a round trip multiplies by the total
grid-point count ``M = M_x M_y M_z``. That factor is folded into the per-mode scalar, so
the projection applied in place on the spectrum is

```math
\hat{\boldsymbol{u}}(\boldsymbol{k}) \leftarrow
\frac{1}{\mu k^2 M}
\left(\hat{\boldsymbol{f}}(\boldsymbol{k})
- \frac{\boldsymbol{k}(\boldsymbol{k}\cdot\hat{\boldsymbol{f}}(\boldsymbol{k}))}{k^2}\right),
\qquad
\hat{\boldsymbol{u}}(\boldsymbol{0}) = \boldsymbol{0}.
```

## Wavenumber layout

The r2c transform stores only the non-negative half of the leading axis, so the spectrum
has shape ``(M_x/2 + 1, M_y, M_z)``. The per-axis wavenumbers are

```math
k_x[i_x] = \frac{2\pi}{L_x}(i_x - 1),
\qquad
k_y[i_y] = \frac{2\pi}{L_y}
\begin{cases} i_y - 1, & i_y \leq M_y/2 + 1, \\ i_y - 1 - M_y, & \text{otherwise}, \end{cases}
```

and ``k_z`` analogous to ``k_y`` — the FFT wrap-around convention that places the
negative frequencies in the upper half of each full axis. The zero mode sits at the
leading index, where the solve applies the gauge fix as a guarded write of zero (no
``1/0`` is ever formed).

The discrete operator is exact in the trigonometric-polynomial basis the FFT represents:
the map from the force density to the fluid velocity is the discrete triply-periodic
Stokes solution at the chosen grid resolution, and the solved velocity field is what the
[interpolation](spreading-interpolation.md) step averages back to the particles.
