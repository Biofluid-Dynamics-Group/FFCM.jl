# The pairwise correction

Step 6 of the pipeline (Su & Keaveny 2024, §4) removes the error the wider spreading
kernel introduced. The [splitting](overview.md)
``\mathcal{M}^{\mathcal{V}\mathcal{F}} = \tilde{\mathcal{M}}^{\mathcal{V}\mathcal{F}} +
(\mathcal{M}^{\mathcal{V}\mathcal{F}} - \tilde{\mathcal{M}}^{\mathcal{V}\mathcal{F}})``
leaves a difference operator that decays like a Gaussian in the pair separation, so it
is applied as a sparse pairwise sum over neighbours within the cutoff ``R_c``, found
through the [cell list](cell-lists.md).

## A difference of two FCM mobilities

The FCM pairwise mobility relating the force on particle ``m`` to the velocity of
particle ``n`` is the regularised Stokeslet at the ``\sqrt{2}``-scaled width,

```math
\boldsymbol{M}^{\mathcal{V}\mathcal{F}}_{nm}
= \boldsymbol{S}(\boldsymbol{Y}_n - \boldsymbol{Y}_m;\ \sigma\sqrt{2})
\qquad (\text{§2, equation (14)}),
```

and the modified-kernel mobility expands as

```math
\tilde{\boldsymbol{M}}^{\mathcal{V}\mathcal{F}}_{nm}
= \boldsymbol{S}(\Sigma\sqrt{2})
+ (\sigma^2 - \Sigma^2)\,\boldsymbol{Q}(\Sigma\sqrt{2})
+ \frac{(\sigma^2 - \Sigma^2)^2}{4}\,\boldsymbol{T}(\Sigma\sqrt{2})
\qquad (\text{§3, equation (29)}),
```

with ``\boldsymbol{S} = \boldsymbol{S}^{(1)} + \boldsymbol{S}^{(2)} + \boldsymbol{S}^{(3)}``
(§2, equations (8)–(10)), ``\boldsymbol{Q} = \boldsymbol{Q}^{(1)} + \boldsymbol{Q}^{(2)}``
(§2, equations (16)–(17)), and ``\boldsymbol{T}`` (§3, equation (30)). Their difference
is the correction. The package evaluates the paper's regrouped closed form (§3, equation
(31)) directly: the ``(\boldsymbol{G} + \sigma^2\Delta\boldsymbol{G})`` grouping — with
``\boldsymbol{G}`` the Stokeslet (Oseen tensor, §2, equation (11)) and
``\Delta\boldsymbol{G}`` its Laplacian — analytically pre-cancels the ``\operatorname{erf}``
of ``\boldsymbol{Q}^{(1)}``, so only ``\boldsymbol{Q}^{(2)}`` and ``\boldsymbol{T}``
(both Gaussian-only) survive as explicit terms.

## The two-scalar collapse

Write ``\boldsymbol{x} = \boldsymbol{Y}_n - \boldsymbol{Y}_m``,
``r = \lVert\boldsymbol{x}\rVert``, and the ``\sqrt{2}``-scaled Gaussian
``\Delta_w \equiv \Delta(\boldsymbol{x}; w\sqrt{2}) = (4\pi w^2)^{-3/2} e^{-r^2/4w^2}``.
Every constituent block is isotropic in the same two tensors, ``\boldsymbol{I}`` and the
un-normalised ``\boldsymbol{x} \otimes \boldsymbol{x}``:

| Block | coefficient of ``\boldsymbol{I}`` | coefficient of ``\boldsymbol{x} \otimes \boldsymbol{x}`` |
|---|---|---|
| ``\boldsymbol{G}`` | ``\dfrac{1}{8\pi\mu r}`` | ``\dfrac{1}{8\pi\mu r^3}`` |
| ``\sigma^2\Delta\boldsymbol{G}`` | ``\dfrac{\sigma^2}{4\pi\mu r^3}`` | ``-\dfrac{3\sigma^2}{4\pi\mu r^5}`` |
| ``\boldsymbol{S}^{(3)}(w\sqrt{2})`` | ``-\dfrac{2w^4}{\mu r^2}\Delta_w`` | ``+\dfrac{6w^4}{\mu r^4}\Delta_w`` |
| ``\boldsymbol{Q}^{(2)}(\Sigma\sqrt{2})`` | ``-\dfrac{1}{\mu}\left(1+\dfrac{2\Sigma^2}{r^2}\right)\Delta_\Sigma`` | ``+\dfrac{1}{\mu r^2}\left(1+\dfrac{6\Sigma^2}{r^2}\right)\Delta_\Sigma`` |
| ``\boldsymbol{T}(\Sigma\sqrt{2})`` | ``\dfrac{1}{2\mu\Sigma^2}\left(2-\dfrac{r^2}{2\Sigma^2}\right)\Delta_\Sigma`` | ``\dfrac{1}{4\mu\Sigma^4}\Delta_\Sigma`` |

so the whole correction collapses to two pair scalars,

```math
\boldsymbol{M}_{nm} - \tilde{\boldsymbol{M}}_{nm}
= A(r)\,\boldsymbol{I} + B(r)\,\boldsymbol{x} \otimes \boldsymbol{x}.
```

Writing ``\mathrm{erf}_{2w} = \operatorname{erf}(r/2w)``,

```math
A(r) = (\mathrm{erf}_{2\sigma}-\mathrm{erf}_{2\Sigma})
\left(\frac{1}{8\pi\mu r}+\frac{\sigma^2}{4\pi\mu r^3}\right)
- \frac{2\sigma^4}{\mu r^2}\Delta_\sigma
+ \left[\frac{2\Sigma^4}{\mu r^2}
+ \frac{\sigma^2-\Sigma^2}{\mu}\left(1+\frac{2\Sigma^2}{r^2}\right)
- \frac{(\sigma^2-\Sigma^2)^2}{4}\cdot\frac{2-r^2/2\Sigma^2}{2\mu\Sigma^2}\right]
\Delta_\Sigma ,
```

```math
B(r) = (\mathrm{erf}_{2\sigma}-\mathrm{erf}_{2\Sigma})
\left(\frac{1}{8\pi\mu r^3}-\frac{3\sigma^2}{4\pi\mu r^5}\right)
+ \frac{6\sigma^4}{\mu r^4}\Delta_\sigma
+ \left[-\frac{6\Sigma^4}{\mu r^4}
- \frac{\sigma^2-\Sigma^2}{\mu r^2}\left(1+\frac{6\Sigma^2}{r^2}\right)
- \frac{(\sigma^2-\Sigma^2)^2}{4}\cdot\frac{1}{4\mu\Sigma^4}\right]
\Delta_\Sigma .
```

The velocity correction to particle ``n`` from the force on a neighbour ``m`` is then

```math
\Delta\boldsymbol{V}_n \mathrel{+}= A(r)\,\boldsymbol{F}_m
+ B(r)\,(\boldsymbol{x}\cdot\boldsymbol{F}_m)\,\boldsymbol{x} .
```

Collapsing to the two scalars replaces a dense ``3\times3`` matrix–vector product with
two scalar evaluations, and the ``\Sigma\sqrt{2}`` ``\operatorname{erf}`` and Gaussian
are shared across the ``\boldsymbol{Q}^{(2)}`` and ``\boldsymbol{T}`` terms, so only two
``\operatorname{erf}`` and two ``\exp`` are evaluated per interaction.

## The erf argument in equation (31)

The printed equation (31) shows the factor
``\operatorname{erf}(r/\sigma\sqrt{2}) - \operatorname{erf}(r/\Sigma\sqrt{2})``, but the
correct argument is ``\operatorname{erf}(r/2\sigma) - \operatorname{erf}(r/2\Sigma)``:
the printed equation carries a typo in the ``\operatorname{erf}`` argument. To see it,
the regularised Stokeslet is
``\boldsymbol{S}^{(1)}(\boldsymbol{x}; \sigma) = \operatorname{erf}(r/\sigma\sqrt{2})\,\boldsymbol{G}``
(equation (12)), and the pairwise mobility evaluates it at the ``\sqrt{2}``-scaled
width (equation (14)), so ``\boldsymbol{S}^{(1)}`` inside
``\boldsymbol{M}^{\mathcal{V}\mathcal{F}}`` carries ``\operatorname{erf}(r/2\sigma)``.
Collecting the ``\Delta\boldsymbol{G}`` terms from
``\boldsymbol{S}^{(2)}(\sigma\sqrt{2}) - \boldsymbol{S}^{(2)}(\Sigma\sqrt{2})
- (\sigma^2-\Sigma^2)\boldsymbol{Q}^{(1)}(\Sigma\sqrt{2})`` then yields the coefficient
``\sigma^2(\mathrm{erf}_{2\sigma} - \mathrm{erf}_{2\Sigma})``, fixing the argument as
``r/2w``. The package implements ``r/2w``; the test suite pins it against an
algebraically independent oracle (the difference of the full mobility expansions), which
would fail under the printed argument.

## The self term

At ``r = 0`` the ``\boldsymbol{x} \otimes \boldsymbol{x}`` part vanishes and the
correction has the well-defined diagonal limit ``\delta\,\boldsymbol{I}`` with

```math
\delta = \frac{1}{6\pi\mu a} - \frac{1}{6\pi\mu(\Sigma\sqrt{\pi})}
+ \frac{\sigma^2-\Sigma^2}{12\mu(\Sigma\sqrt{\pi})^3}
- \frac{(\sigma^2-\Sigma^2)^2}{32\mu\Sigma^5\pi^{3/2}} ,
```

added to every particle independent of its neighbours
(``\boldsymbol{V}_n \mathrel{+}= \delta\boldsymbol{F}_n``). It is what makes the
single-sphere self-mobility come out ``\Sigma``-independent end to end.

## Symmetry, the Σ = σ limit, and periodicity

Because ``A`` and ``B`` depend only on ``r`` and
``A\boldsymbol{I} + B\,\boldsymbol{x} \otimes \boldsymbol{x}`` is a symmetric tensor,
the pair correction is self-adjoint, preserving the
[symmetric positive-definite structure](@ref "Symmetric positive-definiteness") of
the mobility. At ``\Sigma = \sigma`` every Gaussian term and the ``\operatorname{erf}``
difference vanish and ``\delta = 0``: the correction is identically zero, consistent
with ``\tilde{\mathcal{M}}^{\mathcal{V}\mathcal{F}} = \mathcal{M}^{\mathcal{V}\mathcal{F}}``
there.

Under periodicity each pair interacts through its nearest image only: the separation is
reduced to ``[-L_i/2, L_i/2]`` per axis by ``x_i - L_i\operatorname{round}(x_i/L_i)``
(round-to-nearest-ties-to-even, which makes the reduction exactly antisymmetric and the
pair correction exactly symmetric), and the construction-time precondition
``R_c \leq \min(L)/2`` guarantees a single relevant image and excludes any particle
correcting against its own periodic copy.
