using Test
using FFCM
using StructArrays
using StaticArrays

# Boilerplate inputs that satisfy every constructor invariant unless a test
# overrides one of them.
const _DEFAULT_KWARGS = (;
    L = (4.0, 4.0, 4.0),
    R_c = 1.0,
    N = 10,
    Σ_over_σ = 2.0,
    num_grid_points = (Int32(8), Int32(8), Int32(8)),
    M_G = 8,
    μ = 1.0,
)

@testset "Hydrodynamic radius defaults to one (unit-radius convention)" begin
    config = FFCMConfig{Float64}(; _DEFAULT_KWARGS...)
    @test config.a == 1.0
end

@testset "Non-unit hydrodynamic radius is not yet supported" begin
    @test_throws ArgumentError FFCMConfig{Float64}(;
        _DEFAULT_KWARGS..., a = 0.5,
    )
end

@testset "FCM envelope width σ derives from a via the Stokes-drag relation a = σ√π" begin
    config = FFCMConfig{Float64}(; _DEFAULT_KWARGS...)
    @test config.σ ≈ 1.0 / sqrt(π)
end

@testset "Modified-kernel width Σ = (Σ/σ)·σ from the user-supplied ratio (paper §5)" begin
    config = FFCMConfig{Float64}(; _DEFAULT_KWARGS..., Σ_over_σ = 2.5)
    @test config.Σ ≈ 2.5 * config.σ
end

@testset "Σ/σ ≥ 1 is required; the equality case is the standard-FCM degenerate limit" begin
    # Degenerate limit must be admissible.
    config = FFCMConfig{Float64}(; _DEFAULT_KWARGS..., Σ_over_σ = 1.0)
    @test config.Σ ≈ config.σ
    # Below 1 is rejected (paper requires Σ > σ for the speed-up).
    @test_throws ArgumentError FFCMConfig{Float64}(;
        _DEFAULT_KWARGS..., Σ_over_σ = 0.5,
    )
end

@testset "Grid spacing h = L_i / M_i must be isotropic across axes (paper §3 assumption)" begin
    config = FFCMConfig{Float64}(; _DEFAULT_KWARGS...)
    @test config.h ≈ 4.0 / 8
    @test config.inv_h ≈ 1 / config.h

    # Anisotropy in L with isotropy-violating M is rejected.
    @test_throws ArgumentError FFCMConfig{Float64}(;
        _DEFAULT_KWARGS...,
        L = (4.0, 6.0, 8.0),
        num_grid_points = (Int32(8), Int32(8), Int32(8)),
    )
end

@testset "Anisotropic L is allowed when M restores per-axis h equality" begin
    config = FFCMConfig{Float64}(;
        _DEFAULT_KWARGS...,
        L = (4.0, 8.0, 12.0),
        num_grid_points = (Int32(8), Int32(16), Int32(24)),
    )
    @test config.h ≈ 0.5
end

@testset "Kernel grid support M_G is at least two grid points per axis" begin
    @test_throws ArgumentError FFCMConfig{Float64}(;
        _DEFAULT_KWARGS..., M_G = 1,
    )
    @test_throws ArgumentError FFCMConfig{Float64}(;
        _DEFAULT_KWARGS..., M_G = 0,
    )
    config = FFCMConfig{Float64}(; _DEFAULT_KWARGS..., M_G = 2)
    @test config.M_G == Int32(2)
end

@testset "Kernel support M_G must not exceed the grid on any axis" begin
    # A stencil wider than the box would wrap distinct stencil points onto the
    # same grid point (mod M), breaking the spread @simd independence and
    # double-weighting the interpolation adjoint. The default grid is 8³.
    @test_throws ArgumentError FFCMConfig{Float64}(;
        _DEFAULT_KWARGS..., M_G = 10,
    )
    # The smallest grid axis is the binding bound for an anisotropic grid.
    @test_throws ArgumentError FFCMConfig{Float64}(;
        _DEFAULT_KWARGS...,
        L = (4.0, 8.0, 8.0),
        num_grid_points = (Int32(8), Int32(16), Int32(16)),
        M_G = 10,
    )
    # Equality M_G == min(num_grid_points) is admissible (one full period of
    # distinct wrapped indices).
    config = FFCMConfig{Float64}(; _DEFAULT_KWARGS..., M_G = 8)
    @test config.M_G == Int32(8)
end

@testset "Per-axis grid dimension counts must each be at least one" begin
    @test_throws ArgumentError FFCMConfig{Float64}(;
        _DEFAULT_KWARGS...,
        L = (4.0, 4.0, 4.0),
        num_grid_points = (Int32(0), Int32(8), Int32(8)),
    )
end

@testset "Force grid is an SoA StructArray over the (M_x, M_y, M_z) grid" begin
    M = (Int32(8), Int32(12), Int32(16))
    L = (4.0, 6.0, 8.0)
    config = FFCMConfig{Float64}(;
        _DEFAULT_KWARGS...,
        L = L,
        num_grid_points = M,
    )
    @test config.force_density isa StructArray
    @test size(config.force_density) == (8, 12, 16)
    @test eltype(config.force_density) == SVector{3, Float64}
    components = StructArrays.components(config.force_density)
    @test length(components) == 3
    @test all(c -> c isa Array{Float64, 3}, components)
    @test all(c -> size(c) == (8, 12, 16), components)
end

@testset "Per-particle stencil scratch vectors are sized to the kernel support M_G" begin
    # Grid bumped to 16³ so M_G = 10 stays within the per-axis grid bound; this
    # test pins the scratch-vector sizing, which is independent of the grid.
    config = FFCMConfig{Float64}(;
        _DEFAULT_KWARGS...,
        num_grid_points = (Int32(16), Int32(16), Int32(16)),
        M_G = 10,
    )
    @test length(config.gaussian_x) == 10
    @test length(config.gaussian_y) == 10
    @test length(config.gaussian_z) == 10
    @test length(config.r²_x) == 10
    @test length(config.r²_y) == 10
    @test length(config.r²_z) == 10
    @test length(config.idx_x) == 10
    @test length(config.idx_y) == 10
    @test length(config.idx_z) == 10
end

@testset "Cold-path validation works for Float32 as well as Float64" begin
    config = FFCMConfig{Float32}(;
        L = (4.0f0, 4.0f0, 4.0f0),
        R_c = 1.0f0,
        N = 10,
        Σ_over_σ = 2.0f0,
        num_grid_points = (Int32(8), Int32(8), Int32(8)),
        M_G = 8,
        μ = 1.0f0,
    )
    @test config.σ ≈ 1.0f0 / sqrt(Float32(π))
    @test config.Σ ≈ 2.0f0 * config.σ
    @test config.h ≈ 0.5f0
    @test eltype(config.force_density) == SVector{3, Float32}
end
