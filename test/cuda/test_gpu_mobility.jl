using Test
using FFCM
using CUDA
using Random: Xoshiro
using LinearAlgebra: mul!

# CPU↔CUDA parity for the assembled operator `mobility!` (Su & Keaveny 2024, §3-§4): the whole
# six-step pipeline plus the host↔device staging seam. Runs only where the kernels load
# (guarded in runtests.jl). The caller passes ordinary host 3xN arrays to both backends; a GPU
# config uploads the inputs, runs the device-resident pipeline, and downloads the velocities,
# so identical host Y/F must yield velocities matching the CPU backend to round-off (the atomic
# spread/correction and the non-stable device sort fix the result but not the summation order,
# and the corrected velocity is invariant to the unspecified intra-cell order). This is the
# end-to-end parity that composes the per-step parity tests.

# Random positions filling the periodic box with a zero-mean random force pattern. A full-box
# cloud spreads particles across many cells, exercising the device sort, neighbour map, and
# pairwise correction together.
function _random_cloud(::Type{T}, N, L, rng) where {T}
    Y = Matrix{T}(undef, 3, N)
    F = Matrix{T}(undef, 3, N)
    for n in 1:N, i in 1:3
        Y[i, n] = L[i] * rand(rng, T)
        F[i, n] = rand(rng, T) - T(0.5)
    end
    return Y, F
end

@testset "GPU assembled mobility matches the CPU backend" begin
    for T in (Float32, Float64)
        L = (T(8), T(8), T(8))
        rng = Xoshiro(0xC7A0 + (T === Float64 ? 1 : 0))

        # (particles, positions/forces) for the three regimes: an isolated sphere (no
        # correction), a tight cluster (many pairs inside R_c), and a box-filling cloud.
        cases = (
            ("single sphere", 1, (T[L[1] / 2; L[2] / 2; L[3] / 2;;], T[1; 0; 0;;])),
            ("clustered cloud (correction fires)", 64, _clustered_cloud(T, 64)),
            ("random cloud", 256, _random_cloud(T, 256, L, rng)),
        )
        for (name, N, (Y, F)) in cases
            @testset "$name" begin
                cpu = _standard_test_config(T; N = N, L = L)
                gpu = _gpu_config(T; N = N, L = L)
                V_cpu = Matrix{T}(undef, 3, N)
                V_gpu = Matrix{T}(undef, 3, N)
                mobility!(V_cpu, cpu, Y, F)
                mobility!(V_gpu, gpu, Y, F)
                @test _cpu_gpu_isapprox(
                    V_cpu, V_gpu; rtol = sqrt(eps(T)), atol = _near_zero_atol(T),
                )
            end
        end
    end
end

@testset "GPU FFCMMobility / mul! matches the CPU backend" begin
    # The matrix-free operator keeps host scratch and drives the device pipeline unchanged:
    # the host↔device traffic is hidden below the LinearAlgebra interface.
    T = Float32
    N = 128
    L = (T(8), T(8), T(8))
    rng = Xoshiro(0xC7A2)
    Y, F = _random_cloud(T, N, L, rng)

    cpu = _standard_test_config(T; N = N, L = L)
    gpu = _gpu_config(T; N = N, L = L)
    M_cpu = FFCMMobility(cpu, Y)
    M_gpu = FFCMMobility(gpu, Y)

    # Out-of-place 3xN apply.
    @test _cpu_gpu_isapprox(
        M_cpu * F, M_gpu * F; rtol = sqrt(eps(T)), atol = _near_zero_atol(T),
    )

    # Flat-vector 3-arg mul! (the iterative-solver entry point).
    f = vec(F)
    v_cpu = similar(f)
    v_gpu = similar(f)
    mul!(v_cpu, M_cpu, f)
    mul!(v_gpu, M_gpu, f)
    @test _cpu_gpu_isapprox(v_cpu, v_gpu; rtol = sqrt(eps(T)), atol = _near_zero_atol(T))
end

@testset "GPU assembled mobility is allocation-free" begin
    T = Float32
    N = 256
    L = (T(8), T(8), T(8))
    rng = Xoshiro(0xC7A11)
    Y, F = _random_cloud(T, N, L, rng)
    gpu = _gpu_config(T; N = N, L = L)
    V = Matrix{T}(undef, 3, N)

    # Warm up the kernels (first launch JIT-compiles) before measuring.
    mobility!(V, gpu, Y, F)
    @test CUDA.@allocated(mobility!(V, gpu, Y, F)) == 0
end
