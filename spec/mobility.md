# Mobility operator

The assembled Fast FCM mobility operator (Su & Keaveny 2024, §3, equation (19)).

## What the operator computes

The mobility operator $\mathcal{M}^{\mathcal{V}\mathcal{F}}$ maps the $N$ forces
$\mathcal{F}$ localised at positions $\mathcal{Y}$ to the velocities
$\mathcal{V}$ they induce through the triply-periodic Stokes flow (paper §2,
`outline.tex:163`). Fast FCM evaluates it by the splitting
$\mathcal{M} = \widetilde{\mathcal{M}} + (\mathcal{M} - \widetilde{\mathcal{M}})$
(paper §4, `outline.tex:495`): the grid steps 3–5 compute the smooth part
$\widetilde{\mathcal{M}}$ at the modified-kernel width $\Sigma$, and step 6 adds
the real-space correction $\mathcal{M} - \widetilde{\mathcal{M}}$, recovering the
$\sigma$-regularised result independent of $\Sigma$.

## `mobility!(V, config, Y, F)`

The hot path. `Y` and `F` are caller-owned `3×N` matrices (column `n` is
particle `n`, row `i` is Cartesian axis $i \in \{x, y, z\}$); `V` is a
caller-owned `3×N` output matrix. The call writes the velocities into `V` in the
caller's **original** particle order and returns `V`.

Contract:

- **Reads** `Y`, `F`; **writes** `V` and the `config`-owned scratch buffers.
- **Does not mutate** `Y` or `F`. Positions are folded into the canonical domain
  $[0, L_i)$ into the `config`-owned `Y_wrapped` buffer, never in the caller's
  array (so an operator that closes over a fixed `Y` — see below — never sees its
  positions change underneath it).
- **Allocation-free and type-stable** once `config` is built, like every step it
  composes.

The composition is exactly the seven step calls in order:

```
wrap_positions!(config.Y_wrapped, Y, config.L)   # step 1: fold into [0, L)
assign_cells!(config, config.Y_wrapped)          # step 1: cell hash
sort_particles_by_cell!(config, config.Y_wrapped, F)  # step 2: cell list + gather
spread_forces!(config)                           # step 3: J†  (forces → grid)
stokes_solve!(config)                            # step 4: L⁻¹ (FFT Stokes solve)
interpolate_velocities!(V, config)               # step 5: J   (grid → velocities)
correct_velocities!(V, config)                   # step 6: + (M − M̃) real-space
```

Steps 5 and 6 write `V` in original order (interpolation scatters back through
the step-2 permutation; the correction adds onto the same buffer), so `mobility!`
needs no extra reordering.

### One pipeline per call

`mobility!` rebuilds the cell list (wrap → hash → sort → gather) on every call,
even when the same `Y` is reused across a linear solve. This keeps the public
contract a single `mobility!(V, config, Y, F)` and is cheap: the cell-list build
is $O(N)$, dominated by the spread/interpolate ($O(N\,M_G^3)$) and the FFT
($O(M^3 \log M)$) that must run every call because $F$ changes. Building the cell
list once for a fixed `Y` is a future optimisation, not part of this contract.

## Flat-vector convention

The matrix-free interface marshals between the solver's length-$3N$ vectors and
the `3×N` matrices `mobility!` uses. The convention is the column-major `vec` of
a `3×N` matrix: entry $3(n-1) + i$ holds axis $i \in \{1,2,3\}$ of particle $n$.
This is the natural `reshape`, so a caller may equivalently view a velocity
vector as `reshape(v, 3, N)`.

## `FFCMMobility(config, Y)`

A thin operator wrapper closing over `config` and a fixed position matrix `Y`,
implementing the `LinearAlgebra` operator interface so the mobility action drops
straight into `IterativeSolvers.gmres!`, `KrylovKit.linsolve`, or any
`mul!`-based solver without a call-site adapter (`CLAUDE.md` §3).

- `size(M) == (3N, 3N)`, `eltype(M) == T`.
- `mul!(v, M, f)` — applies the mobility once: marshals `f` into a `3×N` scratch,
  runs `mobility!`, marshals the result into `v`. Allocation-free (the flat↔matrix
  copies go through `copyto!`, which respects the column-major convention without
  a `reshape` allocation).
- `mul!(v, M, f, α, β)` — the five-argument form `v .= α·(M·f) + β·v`, with the
  $\beta = 0$ case overwriting `v` (the prior contents are ignored, so an
  uninitialised `v` is allowed). Required by several solvers.

The operator stores its own `3×N` input/output scratch so the marshalling never
allocates on the hot path.

## Symmetry / positive-definiteness

The assembled mobility is symmetric positive-definite: interpolation is the exact
discrete adjoint of spreading (`spec/interpolation.md`), the Stokes solve is
self-adjoint, and the real-space correction is a symmetric pair tensor
(`spec/pairwise-correction.md`). Hence
$\mathcal{G} \cdot (\mathcal{M}\,\mathcal{F}) = \mathcal{F} \cdot (\mathcal{M}\,\mathcal{G})$
for any two force sets, and $\mathcal{F} \cdot (\mathcal{M}\,\mathcal{F}) > 0$ —
the property that makes the operator usable in a conjugate-gradient solve.
