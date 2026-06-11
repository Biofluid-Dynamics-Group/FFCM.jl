using Test
using Random: Xoshiro
using FFCM
using FFCM: _build_cell_list_kernel!, _gather_particles_kernel!,
    sort_particles_by_cell!, wrap_positions!, assign_cells!

@testset "Sorting orders particles by ascending cell index" begin
    # Four particles with out-of-order hashes in a 3-cell layout. After the
    # counting sort, reading the hashes in sorted-slot order must be
    # non-decreasing.
    cell_hash = Int32[2, 0, 2, 1]
    original_index = Vector{Int32}(undef, 4)
    cell_start = Vector{Int32}(undef, 3)
    cell_end = Vector{Int32}(undef, 3)
    counting_sort_scratch = Vector{Int32}(undef, 3)
    _build_cell_list_kernel!(
        original_index, cell_start, cell_end, counting_sort_scratch, cell_hash,
    )
    sorted_hashes = cell_hash[original_index]
    @test issorted(sorted_hashes)
    @test sort(original_index) == Int32[1, 2, 3, 4]  # a true permutation
end

@testset "Each cell's start and end index bracket exactly its particles" begin
    cell_hash = Int32[2, 0, 2, 1]
    original_index = Vector{Int32}(undef, 4)
    cell_start = Vector{Int32}(undef, 3)
    cell_end = Vector{Int32}(undef, 3)
    counting_sort_scratch = Vector{Int32}(undef, 3)
    _build_cell_list_kernel!(
        original_index, cell_start, cell_end, counting_sort_scratch, cell_hash,
    )
    # Every slot in a cell's range belongs to that cell, and the ranges
    # partition the sorted slots 1:N with no gaps or overlaps.
    covered = Int32[]
    for c in 0:2
        for s in cell_start[c + 1]:cell_end[c + 1]
            @test cell_hash[original_index[s]] == c
            push!(covered, s)
        end
    end
    @test sort(covered) == Int32[1, 2, 3, 4]
end

@testset "Empty cells have an empty index range" begin
    # Cell 1 holds no particles in this layout.
    cell_hash = Int32[0, 2, 0, 2]
    original_index = Vector{Int32}(undef, 4)
    cell_start = Vector{Int32}(undef, 3)
    cell_end = Vector{Int32}(undef, 3)
    counting_sort_scratch = Vector{Int32}(undef, 3)
    _build_cell_list_kernel!(
        original_index, cell_start, cell_end, counting_sort_scratch, cell_hash,
    )
    @test cell_end[2] < cell_start[2]               # cell 1: empty range
    @test isempty(cell_start[2]:cell_end[2])
end

@testset "Particles in the same cell keep their original relative order" begin
    # Particles 1, 3, 5 all land in cell 0; the stable sort must list them
    # in ascending original index within that cell.
    cell_hash = Int32[0, 1, 0, 1, 0]
    original_index = Vector{Int32}(undef, 5)
    cell_start = Vector{Int32}(undef, 2)
    cell_end = Vector{Int32}(undef, 2)
    counting_sort_scratch = Vector{Int32}(undef, 2)
    _build_cell_list_kernel!(
        original_index, cell_start, cell_end, counting_sort_scratch, cell_hash,
    )
    @test original_index[cell_start[1]:cell_end[1]] == Int32[1, 3, 5]
    @test original_index[cell_start[2]:cell_end[2]] == Int32[2, 4]
end

@testset "Sorted positions and forces are the gather under the permutation" begin
    for T in (Float32, Float64)
        N = 6
        rng = Xoshiro(2024)
        Y = rand(rng, T, 3, N)
        F = rand(rng, T, 3, N)
        original_index = Int32[4, 1, 6, 2, 5, 3]
        Y_sorted = Matrix{T}(undef, 3, N)
        F_sorted = Matrix{T}(undef, 3, N)
        _gather_particles_kernel!(Y_sorted, F_sorted, Y, F, original_index)
        @test Y_sorted == Y[:, original_index]
        @test F_sorted == F[:, original_index]
    end
end

@testset "Wrapping, hashing, then sorting groups a random cloud by cell" begin
    for T in (Float32, Float64)
        N = 1000
        L = (T(4), T(6), T(8))
        config = FFCMConfig{T}(;
            L = L, R_c = T(1), N = N, _fcm_grid_kwargs(L)...,
        )
        rng = Xoshiro(2024)
        Y = [config.L[i] * rand(rng, T) for i in 1:3, _ in 1:N]
        F = rand(rng, T, 3, N)
        wrap_positions!(Y, config.L)
        assign_cells!(config, Y)
        sort_particles_by_cell!(config, Y, F)
        # Sorted hashes are non-decreasing, and each cell's range holds
        # exactly the particles whose hash equals that cell.
        @test issorted(config.cell_hash[config.original_index])
        total = length(config.cell_start)
        covered = 0
        membership_consistent = true
        for c in 0:(total - 1)
            for s in config.cell_start[c + 1]:config.cell_end[c + 1]
                membership_consistent &=
                    config.cell_hash[config.original_index[s]] == c
                covered += 1
            end
        end
        @test membership_consistent
        @test covered == N
        # The gathered data matches the permutation.
        @test config.Y_sorted == Y[:, config.original_index]
        @test config.F_sorted == F[:, config.original_index]
    end
end
