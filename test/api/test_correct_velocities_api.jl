using Test
using BenchmarkTools
using ForceCouplingMethod
using ForceCouplingMethod: correct_velocities!, wrap_positions!, assign_cells!,
    sort_particles_by_cell!

# Type stability and zero allocation are load-bearing for the hot path: an
# abstract return type would force dynamic dispatch, and a single heap
# allocation per call would dominate the per-iteration cost in the downstream
# mobility solver. The checks target the step entry point; its kernel and the
# correction scalars are covered transitively (a zero-allocation entry bounds
# its callees, and JET walks inference into them — see test_jet.jl). The
# clustered cloud puts many pairs within R_c so the pair branch runs.
@testset "Pairwise-correction hot-path guarantees" begin
    for T in (Float32, Float64)
        N = 32
        config = _standard_test_config(T; N = N)
        Y, F = _clustered_cloud(T, N)
        wrap_positions!(Y, config.L)
        assign_cells!(config, Y)
        sort_particles_by_cell!(config, Y, F)
        V = zeros(T, 3, N)
        @testset "type-stable ($T)" begin
            @inferred correct_velocities!(V, config)
        end
        @testset "allocation-free ($T)" begin
            @test (@ballocated correct_velocities!($V, $config) samples=1 evals=1) == 0
        end
    end
end
