"""
    mobility!(V, config, Y, F) -> V

Apply the Fast FCM mobility operator `M^VF` (Su & Keaveny 2024, §4): map the `N`
forces `F` localised at positions `Y` to the velocities `V` they induce through
the triply-periodic Stokes flow. `Y`, `F`, and `V` are caller-owned `3×N`
matrices (column `n` is particle `n`, row `i` is Cartesian axis `i`); `V` is
overwritten in the caller's **original** particle order and returned.

`Y` and `F` are read-only — positions are folded into `[0, L)` inside a
`config`-owned buffer, never in the caller's array. The call is the single hot
path of the two-phase API: allocation-free and type-stable once `config` is
built. It composes steps 1–6 (spatial hashing, sorting, spreading, the FFT
Stokes solve, interpolation, and the real-space correction); the cell list is
rebuilt every call.

See `spec/mobility.md`.
"""
function mobility!(
    V::AbstractMatrix{T},
    config::FFCMConfig{T},
    Y::AbstractMatrix{T},
    F::AbstractMatrix{T},
) where {T}
    wrap_positions!(config.Y_wrapped, Y, config.L)
    assign_cells!(config, config.Y_wrapped)
    sort_particles_by_cell!(config, config.Y_wrapped, F)
    spread_forces!(config)
    stokes_solve!(config)
    interpolate_velocities!(V, config)
    correct_velocities!(V, config)
    return V
end

"""
    FFCMMobility(config, Y)

Matrix-free `LinearAlgebra` operator backing the mobility action of `mobility!`
by a fixed position matrix `Y` (a `3×N` matrix) and a `config`. It implements
`size`, `eltype`, and the three- and five-argument `mul!`, so it drops straight
into `IterativeSolvers.gmres!`, `KrylovKit.linsolve`, or any `mul!`-based solver.

The operator works on the column-major flat-vector view of the `3×N`
force/velocity matrices: a length-`3N` vector with entry `3(n-1)+i` holding axis
`i` of particle `n`. It owns `3×N` input/output scratch so `mul!` marshals
between the flat vectors and `mobility!`'s matrices without allocating.

See `spec/mobility.md`.
"""
struct FFCMMobility{T, C}
    config::C
    Y::Matrix{T}
    F_mat::Matrix{T}
    V_mat::Matrix{T}
end

function FFCMMobility(config::FFCMConfig{T}, Y::AbstractMatrix{T}) where {T}
    size(Y, 1) == 3 || throw(ArgumentError("Y must be 3×N; got size $(size(Y))"))
    N = size(Y, 2)
    N == size(config.Y_sorted, 2) || throw(ArgumentError(
        "Y has $(N) particles but config was built for " *
        "$(size(config.Y_sorted, 2)); construct a config with matching N",
    ))
    return FFCMMobility{T, typeof(config)}(
        config, Y, Matrix{T}(undef, 3, N), Matrix{T}(undef, 3, N),
    )
end

Base.size(M::FFCMMobility) = (3 * size(M.Y, 2), 3 * size(M.Y, 2))
Base.size(M::FFCMMobility, d::Integer) = d ≤ 2 ? 3 * size(M.Y, 2) : 1
Base.eltype(::FFCMMobility{T}) where {T} = T

# `copyto!` between a length-3N vector and a 3×N matrix copies in linear (column-
# major) order, which is exactly the flat-vector convention — no `reshape`, no
# allocation.
function mul!(
    v::AbstractVector{T}, M::FFCMMobility{T}, f::AbstractVector{T},
) where {T}
    copyto!(M.F_mat, f)
    mobility!(M.V_mat, M.config, M.Y, M.F_mat)
    copyto!(v, M.V_mat)
    return v
end

function mul!(
    v::AbstractVector{T},
    M::FFCMMobility{T},
    f::AbstractVector{T},
    α::Number,
    β::Number,
) where {T}
    copyto!(M.F_mat, f)
    mobility!(M.V_mat, M.config, M.Y, M.F_mat)
    αT = convert(T, α)
    βT = convert(T, β)
    result = M.V_mat
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
