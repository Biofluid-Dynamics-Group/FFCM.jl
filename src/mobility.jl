"""
    mobility!(V, config, Y, F) -> V

Apply the Fast FCM mobility operator `M^VF` (Su & Keaveny 2024, §4): map the `N` forces `F`
localised at positions `Y` to the velocities `V` they induce through the triply-periodic
Stokes flow, composing steps 1-6 (spatial hashing, sorting, spreading, the FFT Stokes solve,
interpolation, and the real-space correction). The cell list is rebuilt on every call.

# Arguments
- `V::AbstractMatrix{T}`: a caller-owned `3xN` output matrix (column `n` is particle `n`,
  row `i` is Cartesian axis `i`); overwritten in the caller's original particle order.
- `config::FFCMConfig{T}`: the compiled configuration owning every hot-path buffer.
- `Y::AbstractMatrix{T}`: the `3xN` particle positions. Read-only — folded into `[0, L)`
  inside a `config`-owned buffer, never in the caller's array.
- `F::AbstractMatrix{T}`: the `3xN` applied forces. Read-only.

# Returns
- `V`: the same matrix, holding the induced velocities.

# Throws
- `DimensionMismatch`: if `V`, `Y`, or `F` is not `3xN` for the `config`'s `N`.

See `spec/mobility.md`.
"""
function mobility!(
    V::AbstractMatrix{T},
    config::FFCMConfig{T},
    Y::AbstractMatrix{T},
    F::AbstractMatrix{T},
) where {T}
    N = size(config.particles.Y_sorted, 2)
    (size(V) == (3, N) && size(Y) == (3, N) && size(F) == (3, N)) ||
        throw(DimensionMismatch(
            "V, Y, and F must each be 3xN for the config's N = $(N); got " *
            "size(V) = $(size(V)), size(Y) = $(size(Y)), size(F) = $(size(F))",
        ))
    wrap_positions!(config.particles.Y_wrapped, Y, config.L)
    assign_cells!(config, config.particles.Y_wrapped)
    sort_particles_by_cell!(config, config.particles.Y_wrapped, F)
    spread_forces!(config)
    stokes_solve!(config)
    interpolate_velocities!(V, config)
    correct_velocities!(V, config)
    return V
end

"""
    FFCMMobility(config, Y) -> FFCMMobility

Matrix-free `LinearAlgebra` operator backing the mobility action of `mobility!` for a fixed
position matrix `Y` and a `config`. It implements `size`, `eltype`, and the three- and
five-argument `mul!`, so it can be passed to any `mul!`-based iterative solver. It also
declares `issymmetric` and `isposdef` (both `true`; the operator is symmetric
positive-definite by construction), so solvers such as `IterativeSolvers.cg!` dispatch on
them.

For convenience, `M * F` applies the operator out of place to a force matrix `F` in the
natural `3xN` layout, returning a fresh `3xN` velocity matrix (the mirror of `mobility!`).
This is distinct from the `3Nx3N` flat-vector view reported by `size(M)` and applied by
`mul!`.

The operator works on the column-major flat-vector view of the `3xN` force/velocity
matrices: a length-`3N` vector with entry `3(n-1)+i` holding axis `i` of particle `n`. It
owns `3xN` scratch to marshal between the flat vectors and `mobility!`'s matrices.

# Arguments
- `config::FFCMConfig{T}`: the compiled configuration; must be built for the same `N` as `Y`.
- `Y::AbstractMatrix{T}`: the fixed `3xN` particle positions the operator acts at.

# Returns
- `FFCMMobility`: the matrix-free operator.

# Throws
- `ArgumentError`: if `Y` is not `3xN`, or its particle count differs from the `config`'s.

See `spec/mobility.md`.
"""
struct FFCMMobility{T, C}
    config::C
    Y::Matrix{T}
    F_scratch::Matrix{T}
    V_scratch::Matrix{T}
end

function FFCMMobility(config::FFCMConfig{T}, Y::AbstractMatrix{T}) where {T}
    size(Y, 1) == 3 || throw(ArgumentError("Y must be 3xN; got size $(size(Y))"))
    N = size(Y, 2)
    N == size(config.particles.Y_sorted, 2) || throw(ArgumentError(
        "Y has $(N) particles but config was built for " *
        "$(size(config.particles.Y_sorted, 2)); construct a config with matching N",
    ))
    return FFCMMobility{T, typeof(config)}(
        config, Y, Matrix{T}(undef, 3, N), Matrix{T}(undef, 3, N),
    )
end

Base.size(M::FFCMMobility) = (3 * size(M.Y, 2), 3 * size(M.Y, 2))
Base.size(M::FFCMMobility, d::Integer) = d ≤ 2 ? 3 * size(M.Y, 2) : 1
Base.eltype(::FFCMMobility{T}) where {T} = T

# `LinearAlgebra.mul!` overrides; the contract is documented in the
# `FFCMMobility` docstring. `copyto!` between a length-3N vector and a 3xN
# matrix copies in linear (column-major) order, which is exactly the
# flat-vector convention — no `reshape` needed.
function mul!(
    v::AbstractVector{T}, M::FFCMMobility{T}, f::AbstractVector{T},
) where {T}
    copyto!(M.F_scratch, f)
    mobility!(M.V_scratch, M.config, M.Y, M.F_scratch)
    copyto!(v, M.V_scratch)
    return v
end

function mul!(
    v::AbstractVector{T},
    M::FFCMMobility{T},
    f::AbstractVector{T},
    α::Number,
    β::Number,
) where {T}
    copyto!(M.F_scratch, f)
    mobility!(M.V_scratch, M.config, M.Y, M.F_scratch)
    αT = convert(T, α)
    βT = convert(T, β)
    result = M.V_scratch
    if iszero(βT)
        # β = 0 overwrites v, ignoring its (possibly uninitialised) contents.
        @inbounds @simd for k in eachindex(v)
            v[k] = αT * result[k]
        end
    else
        @inbounds @simd for k in eachindex(v)
            v[k] = αT * result[k] + βT * v[k]
        end
    end
    return v
end

# `M * F` in the natural `3xN` layout, mirroring `mobility!` but allocating a
# fresh result. The shape is enforced by `mobility!`'s entry guard. Distinct
# from the `3Nx3N` flat-vector view used by `mul!` (see spec/mobility.md).
function Base.:*(M::FFCMMobility{T}, F::AbstractMatrix{T}) where {T}
    N = size(M.Y, 2)
    return mobility!(Matrix{T}(undef, 3, N), M.config, M.Y, F)
end

# The assembled mobility is symmetric positive-definite by construction (see
# spec/mobility.md); declaring the traits lets generic solvers dispatch on them.
issymmetric(::FFCMMobility) = true
isposdef(::FFCMMobility) = true
