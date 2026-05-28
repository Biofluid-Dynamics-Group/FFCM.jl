using Test
using FFCM
using StructArrays
using StaticArrays

# Boilerplate inputs that satisfy every cycle-1 invariant unless a test
# overrides one of them.
const _DEFAULT_KWARGS = (;
    L = (4.0, 4.0, 4.0),
    R_c = 1.0,
    N = 10,
    Σ_over_σ = 2.0,
    num_grid_points = (Int32(8), Int32(8), Int32(8)),
    M_G = 8,
)

@testset "Hydrodynamic radius defaults to one (CLAUDE.md unit-radius contract)" begin
    config = FFCMConfig{Float64}(; _DEFAULT_KWARGS...)
    @test config.a == 1.0
end

@testset "Non-unit hydrodynamic radius is not yet supported" begin
    @test_throws ErrorException FFCMConfig{Float64}(;
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

@testset "Grid spacing Δx = L_i / M_i must be isotropic across axes (paper §3 assumption)" begin
    config = FFCMConfig{Float64}(; _DEFAULT_KWARGS...)
    @test config.Δx ≈ 4.0 / 8
    @test config.inv_Δx ≈ 1 / config.Δx

    # Anisotropy in L with isotropy-violating M is rejected.
    @test_throws ArgumentError FFCMConfig{Float64}(;
        _DEFAULT_KWARGS...,
        L = (4.0, 6.0, 8.0),
        num_grid_points = (Int32(8), Int32(8), Int32(8)),
    )
end

@testset "Anisotropic L is allowed when M restores per-axis Δx equality" begin
    config = FFCMConfig{Float64}(;
        _DEFAULT_KWARGS...,
        L = (4.0, 8.0, 12.0),
        num_grid_points = (Int32(8), Int32(16), Int32(24)),
    )
    @test config.Δx ≈ 0.5
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
    @test config.force_grid isa StructArray
    @test size(config.force_grid) == (8, 12, 16)
    @test eltype(config.force_grid) == SVector{3, Float64}
    components = StructArrays.components(config.force_grid)
    @test length(components) == 3
    @test all(c -> c isa Array{Float64, 3}, components)
    @test all(c -> size(c) == (8, 12, 16), components)
end

@testset "Per-particle stencil scratch vectors are sized to the kernel support M_G" begin
    config = FFCMConfig{Float64}(; _DEFAULT_KWARGS..., M_G = 10)
    @test length(config.gauss_x) == 10
    @test length(config.gauss_y) == 10
    @test length(config.gauss_z) == 10
    @test length(config.r²_x) == 10
    @test length(config.r²_y) == 10
    @test length(config.r²_z) == 10
    @test length(config.ind_x) == 10
    @test length(config.ind_y) == 10
    @test length(config.ind_z) == 10
end

@testset "Cold-path validation works for Float32 as well as Float64" begin
    config = FFCMConfig{Float32}(;
        L = (4.0f0, 4.0f0, 4.0f0),
        R_c = 1.0f0,
        N = 10,
        Σ_over_σ = 2.0f0,
        num_grid_points = (Int32(8), Int32(8), Int32(8)),
        M_G = 8,
    )
    @test config.σ ≈ 1.0f0 / sqrt(Float32(π))
    @test config.Σ ≈ 2.0f0 * config.σ
    @test config.Δx ≈ 0.5f0
    @test eltype(config.force_grid) == SVector{3, Float32}
end
