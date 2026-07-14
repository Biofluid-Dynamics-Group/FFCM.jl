using Test
using FFCM
using CUDA
using Random: Xoshiro
using FFCM: wrap_positions!, assign_cells!, sort_particles_by_cell!, _build_neighbor_map

# CPU↔CUDA parity for pipeline step 1 (spatial hashing + cell list). Runs only
# where `CUDA.functional()` (guarded in runtests.jl). The GPU cell list is a
# counting sort with a non-stable atomic scatter, so the intra-cell order of the
# permutation is unspecified: per-cell ranges and per-particle hashes are compared
# directly, the permutation by its partition invariants.

@testset "GPU periodic wrap matches the CPU backend" begin
    for T in (Float32, Float64)
        N = 256
        L = (T(8), T(8), T(8))
        rng = Xoshiro(0xC2A0 + (T === Float64 ? 1 : 0))
        # Mid-domain targets shifted by whole boxes: wrapping returns the target,
        # far from any cell boundary, so CPU/GPU `mod` agree to round-off.
        Y = Matrix{T}(undef, 3, N)
        target = Matrix{T}(undef, 3, N)
        for n in 1:N, i in 1:3
            target[i, n] = (T(0.25) + T(0.5) * rand(rng, T)) * L[i]
            Y[i, n] = target[i, n] + T(rand(rng, -2:2)) * L[i]
        end

        cpu = _standard_test_config(T; N = N, L = L)
        gpu = _gpu_config(T; N = N, L = L)
        wrap_positions!(cpu.particles.Y_wrapped, Y, L)
        wrap_positions!(gpu.particles.Y_wrapped, CuArray(Y), L)

        @test _cpu_gpu_isapprox(
            cpu.particles.Y_wrapped, gpu.particles.Y_wrapped;
            rtol = sqrt(eps(T)), atol = _near_zero_atol(T),
        )
        wrapped = Array(gpu.particles.Y_wrapped)
        for i in 1:3
            @test all(>=(zero(T)), wrapped[i, :])
            @test all(<(L[i]), wrapped[i, :])
        end
    end
end

@testset "GPU cell list matches the CPU backend" begin
    for T in (Float32, Float64)
        N = 1000
        L = (T(8), T(8), T(8))
        rng = Xoshiro(0xCE11 + (T === Float64 ? 1 : 0))
        # Positions strictly inside the box (so the wrap is the exact identity and
        # the hashes compare bit-for-bit), with x confined to the lower half so the
        # upper-x cells are guaranteed empty (exercises the empty-range path).
        Y = Matrix{T}(undef, 3, N)
        F = Matrix{T}(undef, 3, N)
        for n in 1:N
            Y[1, n] = rand(rng, T) * (L[1] / T(2))
            Y[2, n] = rand(rng, T) * L[2]
            Y[3, n] = rand(rng, T) * L[3]
            F[1, n] = rand(rng, T) - T(0.5)
            F[2, n] = rand(rng, T) - T(0.5)
            F[3, n] = rand(rng, T) - T(0.5)
        end

        cpu = _standard_test_config(T; N = N, L = L)
        gpu = _gpu_config(T; N = N, L = L)

        wrap_positions!(cpu.particles.Y_wrapped, Y, L)
        assign_cells!(cpu, cpu.particles.Y_wrapped)
        sort_particles_by_cell!(cpu, cpu.particles.Y_wrapped, F)

        Yd, Fd = CuArray(Y), CuArray(F)
        wrap_positions!(gpu.particles.Y_wrapped, Yd, L)
        assign_cells!(gpu, gpu.particles.Y_wrapped)
        sort_particles_by_cell!(gpu, gpu.particles.Y_wrapped, Fd)

        @testset "per-particle hashes and per-cell ranges are identical" begin
            @test Array(gpu.cells.cell_hash) == cpu.cells.cell_hash
            @test Array(gpu.cells.cell_start) == cpu.cells.cell_start
            @test Array(gpu.cells.cell_end) == cpu.cells.cell_end
            # The lower-half-x cloud guarantees at least one empty cell.
            @test any(cpu.cells.cell_end .< cpu.cells.cell_start)
        end

        @testset "permutation is a valid cell grouping" begin
            perm = Array(gpu.cells.original_index)
            @test sort(perm) == collect(Int32(1):Int32(N))
            @test issorted(Array(gpu.cells.cell_hash)[perm])
        end

        @testset "gathered data is consistent with the permutation" begin
            perm = Array(gpu.cells.original_index)
            @test Array(gpu.particles.Y_sorted) == Array(gpu.particles.Y_wrapped)[:, perm]
            @test Array(gpu.particles.F_sorted) == Array(Fd)[:, perm]
        end

        @testset "device neighbour map equals the host build" begin
            @test Array(gpu.cells.neighbor_map) == _build_neighbor_map(gpu.num_cells)
        end
    end
end

@testset "GPU cell list is allocation-free" begin
    T = Float32
    N = 512
    L = (T(8), T(8), T(8))
    gpu = _gpu_config(T; N = N, L = L)
    rng = Xoshiro(0xA110C)
    Y = Matrix{T}(undef, 3, N)
    F = Matrix{T}(undef, 3, N)
    for n in 1:N
        Y[1, n] = rand(rng, T) * L[1]
        Y[2, n] = rand(rng, T) * L[2]
        Y[3, n] = rand(rng, T) * L[3]
        F[:, n] .= rand(rng, T)
    end
    Yd, Fd = CuArray(Y), CuArray(F)

    # Warm up the kernels (first launch JIT-compiles) before measuring.
    wrap_positions!(gpu.particles.Y_wrapped, Yd, L)
    assign_cells!(gpu, gpu.particles.Y_wrapped)
    sort_particles_by_cell!(gpu, gpu.particles.Y_wrapped, Fd)

    @test CUDA.@allocated(wrap_positions!(gpu.particles.Y_wrapped, Yd, L)) == 0
    @test CUDA.@allocated(assign_cells!(gpu, gpu.particles.Y_wrapped)) == 0
    # The counting-sort prefix sum is allocation-free at this cell count
    # (single-block `accumulate!` scan). A large cell count takes CUDA.jl's
    # multi-block scan, which allocates a transient aggregate buffer — an
    # allocation-free scan for large grids is a deferred, benchmark-gated
    # follow-up.
    @test CUDA.@allocated(
        sort_particles_by_cell!(gpu, gpu.particles.Y_wrapped, Fd)
    ) == 0
end
