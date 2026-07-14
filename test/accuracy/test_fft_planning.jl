using Test
using FFCM

@testset "Self-mobility is independent of the Fourier planner effort" begin
    # FFTW's measuring planner levels execute candidate transforms on the grid
    # buffers during construction, so the *first* mobility! call after a
    # measured-plan construction is the case that would expose planner
    # scribble surviving in a grid buffer. The σ-regularised periodic
    # self-mobility (lattice-sum oracle in test_utilities.jl) pins the
    # physics; the cross-effort comparison pins that the solution depends on
    # the planner only through floating-point ordering inside the transforms.
    for T in (Float32, Float64)
        L = (T(8), T(8), T(8))
        Y = T[L[1] / 2; L[2] / 2; L[3] / 2;;]
        F = T[1; 0; 0;;]
        velocities = Dict{Symbol, T}()
        for fft_planning in (:estimate, :measure)
            config = FFCMConfig{T}(;
                L = L, R_c = T(1), N = 1, kernel_widths_ratio = T(1.25),
                viscosity = T(1), num_grid_points = (Int32(32), Int32(32), Int32(32)),
                M_G = 31, fft_planning = fft_planning,
            )
            V = zeros(T, 3, 1)
            mobility!(V, config, Y, F)
            velocities[fft_planning] = V[1, 1]
        end

        # Grid-resolution / stencil-truncation limited, as in the six-step
        # single-sphere test (same geometry); the Float64 oracle keeps the
        # reference precision above the truncation error at both T.
        reference = _periodic_self_mobility_xx(1 / sqrt(π), 1.0, 8.0)
        @test velocities[:estimate] ≈ T(reference) rtol = T(1e-4)

        # Effort invariance: identical pipeline, plans differing only in the
        # algorithm FFTW selected — round-off-level agreement.
        @test velocities[:measure] ≈ velocities[:estimate] rtol = sqrt(eps(T))
    end
end

@testset "Unknown planner effort is rejected at construction" begin
    for T in (Float32, Float64)
        @test_throws ArgumentError FFCMConfig{T}(;
            L = (T(8), T(8), T(8)), R_c = T(1), N = 1, kernel_widths_ratio = T(2),
            viscosity = T(1), num_grid_points = (Int32(16), Int32(16), Int32(16)),
            M_G = 8, fft_planning = :exhaustive,
        )
    end
end
