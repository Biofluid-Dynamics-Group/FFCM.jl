using Test
using Random: Xoshiro
using FFCM
using LinearAlgebra: dot

# Seeded property tests: the operator identities that define M^VF — linearity
# in F, symmetry, and positive-definiteness (paper §3; spec/mobility.md) — are
# exact algebraic properties of the assembled discrete operator, so they must
# hold for *any* admissible configuration, not just the hand-picked geometries
# of the accuracy tests. Each trial draws a random domain, grid, kernel-width
# ratio, viscosity, particle count, positions (including out-of-domain ones,
# exercising the wrap), and forces from a fixed-seed RNG, so failures are
# reproducible.
#
# Parameter ranges stay inside the paper's calibrated regime (σ/h ≈ 1.13 at
# h = 1/2, Σ/σ ∈ [1, 2.5], M_G up to the grid bound): the identities are
# checked at round-off tolerance, which only holds when the discretisation is
# resolved.
function _random_mobility_setup(rng, ::Type{T}) where {T}
    h = T(0.5)
    M = ntuple(_ -> Int32(rand(rng, (8, 10, 12, 14, 16))), 3)
    L = ntuple(i -> T(M[i]) * h, 3)
    N = rand(rng, 2:8)
    config = FFCMConfig{T}(;
        L = L,
        R_c = T(1),
        N = N,
        kernel_widths_ratio = one(T) + T(1.5) * rand(rng, T),
        viscosity = T(0.5) + T(2) * rand(rng, T),
        num_grid_points = M,
        M_G = rand(rng, 6:Int(minimum(M))),
    )
    # Positions in [-L_i, 2L_i): two thirds of them outside the canonical
    # domain, so mobility! wraps them.
    Y = T[(T(3) * rand(rng, T) - one(T)) * L[i] for i in 1:3, _ in 1:N]
    return config, Y, N
end

@testset "Mobility operator identities hold across random configurations" begin
    for T in (Float32, Float64)
        rng = Xoshiro(2024)
        rtol = sqrt(eps(T))
        atol = _near_zero_atol(T)
        for _ in 1:6
            config, Y, N = _random_mobility_setup(rng, T)
            F = T[T(2) * rand(rng, T) - one(T) for _ in 1:3, _ in 1:N]
            G = T[T(2) * rand(rng, T) - one(T) for _ in 1:3, _ in 1:N]
            a = T(2) * rand(rng, T) - one(T)
            b = T(2) * rand(rng, T) - one(T)

            V_F = zeros(T, 3, N)
            V_G = zeros(T, 3, N)
            V_combined = zeros(T, 3, N)
            mobility!(V_F, config, Y, F)
            mobility!(V_G, config, Y, G)
            mobility!(V_combined, config, Y, a .* F .+ b .* G)

            # Linearity: M(aF + bG) = a⋅MF + b⋅MG.
            @test isapprox(
                V_combined, a .* V_F .+ b .* V_G; rtol = rtol, atol = atol,
            )

            # Symmetry: ⟨G, MF⟩ = ⟨F, MG⟩.
            @test dot(G, V_F) ≈ dot(F, V_G) rtol = rtol atol = atol

            # Positivity: ⟨F, MF⟩ > 0 for F ≠ 0.
            @test dot(F, V_F) > zero(T)
        end
    end
end
