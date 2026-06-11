using Test
using FFCM: wrap_positions!

# The cuFCM reference implementation handles the upper-edge case explicitly
# in `images()` (CUFCM_DATA.cu): after the modulo fold, if x == L, it sets
# x = 0. Our `wrap_positions!` reaches the same outcome through `mod(Y, L)`
# alone, because `mod(L, L) == 0` for the floating-point types we support.
# This test pins parity with the cuFCM behaviour at the upper boundary.
@testset "Position on the upper boundary wraps to the lower edge" begin
    L = (4.0, 6.0, 8.0)
    Y = reshape(Float64[4.0, 6.0, 8.0], 3, 1)
    wrap_positions!(Y, L)
    @test Y[1, 1] == 0.0
    @test Y[2, 1] == 0.0
    @test Y[3, 1] == 0.0
end

@testset "Positions already in the canonical domain are unchanged" begin
    L = (4.0, 6.0, 8.0)
    Y = Float64[1.5 0.0 3.999;
                2.5 5.999 0.0;
                3.5 0.0 7.999]
    expected = copy(Y)
    wrap_positions!(Y, L)
    @test Y == expected
end

@testset "Wrapping is idempotent" begin
    L = (4.0, 6.0, 8.0)
    Y = Float64[-1.3  9.7  -100.0;
                 0.0  7.5    20.0;
                 8.5 -0.1   -8.0]
    wrap_positions!(Y, L)
    once = copy(Y)
    wrap_positions!(Y, L)
    @test Y == once
end

@testset "Translating a position by an integer multiple of L preserves the wrap" begin
    L = (4.0, 6.0, 8.0)
    base = Float64[1.5; 2.5; 3.5;;]
    Y_base = copy(base)
    wrap_positions!(Y_base, L)
    for (k1, k2, k3) in ((1, 0, 0), (0, 1, 0), (0, 0, 1), (-2, 3, -5))
        offset = Float64[k1 * L[1]; k2 * L[2]; k3 * L[3];;]
        Y_shifted = base .+ offset
        wrap_positions!(Y_shifted, L)
        @test Y_shifted ≈ Y_base
    end
end

@testset "Slightly negative positions wrap to just below the upper edge" begin
    # Y_i = -ε for small positive ε should map to L_i - ε, completing the
    # half-open [0, L_i) wrap.
    L = (4.0, 6.0, 8.0)
    ε = 1e-9
    Y = Float64[-ε; -ε; -ε;;]
    wrap_positions!(Y, L)
    @test Y[1, 1] ≈ L[1] - ε
    @test Y[2, 1] ≈ L[2] - ε
    @test Y[3, 1] ≈ L[3] - ε
end
