# Mobility Operator

The assembled Fast FCM mobility operator (Su & Keaveny 2024, §3 and §4).

## Summary

The mobility operator $\mathcal{M}^{\mathcal{V}\mathcal{F}}$ maps the $N$ forces $\mathcal{F}$
localised at positions $\mathcal{Y}$ to the velocities $\mathcal{V}$ they induce through the
triply-periodic Stokes flow (§2, equations (5)–(6)). Fast FCM evaluates it by the splitting
$$
\mathcal{M}^{\mathcal{V}\mathcal{F}}
= \tilde{\mathcal{M}}^{\mathcal{V}\mathcal{F}}
+ \left(\mathcal{M}^{\mathcal{V}\mathcal{F}} - \tilde{\mathcal{M}}^{\mathcal{V}\mathcal{F}}\right)
\qquad (\text{§3, equation (19)}),
$$
in which the spectral steps — spreading, the Stokes solve, and interpolation — compute the
smooth part $\tilde{\mathcal{M}}^{\mathcal{V}\mathcal{F}}$ at the modified-kernel width
$\Sigma$, and the real-space pairwise correction adds
$\mathcal{M}^{\mathcal{V}\mathcal{F}} - \tilde{\mathcal{M}}^{\mathcal{V}\mathcal{F}}$,
recovering the $\sigma$-regularised result independent of $\Sigma$.

In the code, we define
- `mobility!(V, config, Y, F)` — the hot-path application of $\mathcal{M}^{\mathcal{V}\mathcal{F}}$, writing $\mathcal{V}$ into `V`;
- `FFCMMobility` — the matrix-free `LinearAlgebra` operator backing the same action for iterative solvers.

## Method

### The six-step composition

The operator is assembled from the six sub-algorithms of §4, applied in order; step 1
(spatial hashing) is two calls — the position wrap and the cell hash — so the six steps
are seven calls. `mobility!(V, config, Y, F)` runs exactly:

```
wrap_positions!(config.Y_wrapped, Y, config.L)        # step 1: fold into [0, L)
assign_cells!(config, config.Y_wrapped)               # step 1: cell hash
sort_particles_by_cell!(config, config.Y_wrapped, F)  # step 2: cell list + gather
spread_forces!(config)                                # step 3: J̃†  (forces → density)
stokes_solve!(config)                                 # step 4: L⁻¹ (FFT Stokes solve)
interpolate_velocities!(V, config)                    # step 5: J̃   (velocity → particles)
correct_velocities!(V, config)                        # step 6: + (M − M̃) real-space
```

Steps 1–2 build the cell list and gather the forces; steps 3–5 compute
$\tilde{\mathcal{M}}^{\mathcal{V}\mathcal{F}}\mathcal{F}$; step 6 adds the real-space
remainder. Interpolation and the correction both write `V` in the caller's original order —
interpolation scatters back through the step-2 permutation, and the correction adds onto the
same buffer — so `mobility!` needs no extra reordering.

### Symmetric positive-definiteness

The assembled mobility is symmetric positive-definite. Interpolation is the exact discrete
adjoint of spreading ([interpolation.md](interpolation.md)), the Stokes solve is self-adjoint
([stokes-solve.md](stokes-solve.md)), and the real-space correction is a symmetric pair
tensor ([pairwise-correction.md](pairwise-correction.md)). Hence
$\mathcal{G} \cdot (\mathcal{M}\mathcal{F}) = \mathcal{F} \cdot (\mathcal{M}\mathcal{G})$
for any two force sets, and $\mathcal{F} \cdot (\mathcal{M}\mathcal{F}) > 0$ — the property
a conjugate-gradient resistance solve depends on (paper §3.3, "Positive splitting").

### Flat-vector convention

The matrix-free interface marshals between a solver's length-$3N$ vectors and the
$3 \times N$ matrices `mobility!` uses. The convention is the column-major `vec` of a
$3 \times N$ matrix: entry $3(n-1) + i$ holds axis $i \in \{1, 2, 3\}$ of particle $n$. This
is the natural `reshape`, so a velocity vector may equivalently be viewed as
`reshape(v, 3, N)`.

## Contract

### `mobility!(V, config, Y, F)`

- `Y`, `F`, `V` are caller-owned $3 \times N$ matrices (column $n$ is particle $n$, row $i$
  is axis $i$).
- Reads `Y` and `F`; writes `V` in the caller's **original** particle order; returns `V`.
- Does **not** mutate `Y` or `F`: positions are folded into $[0, L_i)$ in the `config`-owned
  buffer `Y_wrapped`, never in the caller's array, so an operator that closes over a fixed
  `Y` never sees its positions change underneath it.
- Allocation-free and type-stable once `config` is built.

### `FFCMMobility(config, Y)`

A thin operator closing over `config` and a fixed position matrix `Y`, implementing the
`LinearAlgebra` operator interface so the action drops straight into `IterativeSolvers.gmres!`,
`KrylovKit.linsolve`, or any `mul!`-based solver without a call-site adapter.

- `size(M) == (3N, 3N)`, `eltype(M) == T`.
- `mul!(v, M, f)` — marshals `f` into a $3 \times N$ scratch, runs `mobility!`, and marshals
  the result into `v`. Allocation-free.
- `mul!(v, M, f, α, β)` — the five-argument form `v .= α·(M·f) + β·v`; the $\beta = 0$ case
  overwrites `v` (its prior, possibly uninitialised, contents are ignored).
- The constructor validates `size(Y, 1) == 3` and that `Y`'s particle count matches the one
  `config` was built for.

## Implementation

`mobility!` is the composition above and nothing more — the seven step calls in order,
each operating on `config`-owned buffers. It is the single hot path of the two-phase API;
all per-call work writes into pre-allocated buffers, so it allocates nothing once `config`
is built.

`FFCMMobility` stores `config`, the fixed `Y`, and its own $3 \times N$ input/output
scratch. `mul!` copies the flat force vector into the scratch with `copyto!`, calls
`mobility!`, and copies the scratch velocity back out — `copyto!` between a length-$3N$
vector and a $3 \times N$ matrix moves data in column-major order, which is exactly the
flat-vector convention, so no `reshape` and no allocation are needed. The five-argument
`mul!` applies the `α`/`β` scaling in a single pass over the output.

| Phase | Allocations | Functions |
|---|---|---|
| Cold | OK | `FFCMConfig(...)` (all earlier steps); `FFCMMobility(config, Y)` allocates its scratch. |
| Hot  | `@ballocated == 0` | `mobility!(V, config, Y, F)`; `mul!(v, M, f)` and `mul!(v, M, f, α, β)`. |

## Performance notes

### One pipeline per call

`mobility!` rebuilds the cell list (wrap → hash → sort → gather) on every call, even when
the same `Y` is reused across a linear solve. This keeps the public contract a single
`mobility!(V, config, Y, F)` and is cheap: the cell-list build is $\mathcal{O}(N)$,
dominated by the spread and interpolation ($\mathcal{O}(NM_G^3)$) and the FFT
($\mathcal{O}(M \log M)$) that must run every call because `F` changes. Building the cell
list once for a fixed `Y` is a future optimisation, not part of this contract.

The matrix-free marshalling is allocation-free: the flat↔matrix copies go through `copyto!`,
which respects the column-major convention without a `reshape` allocation.

## Verification

- `test/test_mobility.jl` — the composition matches the six steps applied by hand; `Y` and
  `F` are not mutated; `V` comes out in the caller's original order; the 3- and 5-argument
  `mul!` agree with `mobility!` and obey the `α`/`β` contract (including $\beta = 0$ on an
  uninitialised `v`); `size` and `eltype` are correct; the `FFCMMobility` constructor rejects
  a non-`3×N` `Y` and an `N` mismatch.
- `test/test_single_sphere_mobility.jl` — the end-to-end accuracy result: the assembled
  operator reproduces the $\sigma$-regularised single-sphere periodic self-mobility
  independent of $\Sigma$, to grid-truncation tolerance.
- `test/test_mobility_inferred.jl` — `@inferred` for `mobility!` and both `mul!` methods,
  `Float32`/`Float64`.
- `test/test_mobility_allocations.jl` — `@ballocated == 0` for `mobility!` and both `mul!`
  methods.
- `test/test_jet.jl` — `JET.@test_call mobility!` over the full call graph.
- `test/test_aqua.jl` — package hygiene.

## Differences from cuFCM

> Comparison against the C++/CUDA reference implementation, kept for validation during
> development and removed once the port is complete.

cuFCM's `FCM_solver` (`cuFCM/src/CUFCM_SOLVER.cu`) orchestrates the same sequence — spatial
hashing, sorting, spreading, the FFT Stokes solve, interpolation, and the real-space
correction — keeping all intermediates resident on the device across the run. This package
composes the same six steps behind the single `mobility!` entry, rebuilding the cell list per
call and marshalling through `FFCMMobility` for the matrix-free `mul!` interface. The
step-by-step algorithmic differences are documented in each step's spec; at the assembled
level the only divergence is the per-call cell-list rebuild versus cuFCM's persistent device
state.
