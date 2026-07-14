# Cell lists

The [pairwise correction](pairwise-correction.md) must find, for every particle, all
neighbours within the cutoff ``R_c``. Steps 1–2 of the pipeline (Su & Keaveny 2024, §4)
make that search ``\mathcal{O}(N)``: the periodic box
``\Omega = [0, L_x) \times [0, L_y) \times [0, L_z)`` is partitioned into rectangular
cells sized so that every neighbour within ``R_c`` of a particle lies in the particle's
own cell or one of its 26 immediate neighbours, and the particle data are sorted so each
cell's particles occupy contiguous memory.

## Cell geometry and the covering guarantee

The number of cells per axis is

```math
m_i = \max\!\left(\left\lfloor \frac{L_i}{R_c} \right\rfloor,\, 3\right)
\qquad (\text{§4, Step 1}).
```

Why the ``3 \times 3 \times 3`` block suffices splits into two regimes, depending on
which term of the ``\max`` is active.

**The box is at least three cutoffs wide on axis ``i``** (``L_i/R_c \geq 3``). Then
``m_i = \lfloor L_i/R_c \rfloor \leq L_i/R_c``, so each cell is at least as wide as the
cutoff, ``L_i/m_i \geq R_c``. Any particle within ``R_c`` of a given particle differs by
at most one cell width per axis, so its cell coordinate differs by at most one, and the
``3 \times 3 \times 3`` block contains the whole ``R_c``-ball.

**The box is narrower than three cutoffs on axis ``i``** (``2 \leq L_i/R_c < 3``, the
range still permitted by the correction's precondition ``R_c \leq \min(L)/2``). The
floor clamps to ``m_i = 3`` and a cell may be narrower than the cutoff, but coverage
holds for a different reason: with only three cells on the axis, the neighbour offsets
``\{-1, 0, +1\} \bmod 3 = \{0, 1, 2\}`` span every cell, so the block sweeps the whole
axis and no neighbour can be missed.

In both regimes the floor ``m_i \geq 3`` also guarantees that, under periodicity, the 26
neighbours are distinct cells, so none is double-counted (with ``m_i = 2`` the offsets
``\{-1, +1\} \bmod 2`` would collide).

## Hashing a position to a cell

Positions are first folded into ``[0, L_i)`` by ``Y_{n,i} \mapsto \operatorname{mod}(Y_{n,i}, L_i)``.
This is the canonical wrap point for the whole pipeline: after it, every later step may
assume ``\boldsymbol{Y}_n \in \Omega``. A wrapped position then maps to the cell
coordinates and linear cell index

```math
c_i = \min\!\left(\left\lfloor \frac{Y_{n,i}}{\text{cell size}_i} \right\rfloor,\, m_i - 1\right),
\qquad
\text{cell index} = c_x + (c_y + c_z m_y)\, m_x
\qquad (\text{§4, equation (68)}),
```

with ``x`` laid out fastest and ``z`` slowest. The ``\min(\cdot, m_i - 1)`` clamp guards
one floating-point corner: if ``Y_{n,i}`` is just below ``L_i`` and the division rounds
up to exactly ``m_i``, the floor would land one past the last cell; the clamp folds it
back. The boundary behaviour is therefore: ``Y_{n,i} = 0`` maps to cell coordinate ``0``,
and ``Y_{n,i} = L_i`` — reachable only by floating-point round-off — maps to
``m_i - 1``.

## Sorting particles by cell

With every particle hashed, the particle data are ordered by cell with a **counting
sort** on the integer key. Because the keys are bounded by the cell count, no comparison
sort is needed, and the per-cell index ranges fall out of the sort itself:

1. **Histogram.** Count how many particles fall in each cell.
2. **Exclusive prefix sum.** Turn the counts into the 1-based inclusive range
   ``\text{cell start}[c] : \text{cell end}[c]`` of sorted slots cell ``c`` will occupy.
   An empty cell gets ``\text{cell end}[c] = \text{cell start}[c] - 1`` — a well-defined
   empty range for any occupancy, so the correction can iterate any cell without a
   guard.
3. **Stable scatter.** Walk the particles in ascending original index and place each in
   the next free slot of its cell, recording the permutation
   ``\text{original index}[s] = n`` from sorted slot ``s`` to original particle ``n``.

A final gather materialises the sorted data,
``\boldsymbol{Y}^{\text{sorted}}_s = \boldsymbol{Y}_{\text{original index}[s]}`` and
likewise for the forces, so particles sharing a cell occupy contiguous columns. That
contiguity is the memory locality the later steps rely on: the
[spread and interpolation](spreading-interpolation.md) touch overlapping grid patches
for consecutive particles, and the [pairwise correction](pairwise-correction.md) walks a
cell by its first and last slot.
