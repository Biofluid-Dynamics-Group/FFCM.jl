# Benchmark suite for the FFCM mobility pipeline. Defines `SUITE` in the
# `BenchmarkGroup` convention consumed by `run.jl`/`judge.jl` (see README.md)
# and by any external harness that loads `benchmark/benchmarks.jl`.

using BenchmarkTools
using Random: Xoshiro
using FFCM
using FFCM:
    wrap_positions!,
    assign_cells!,
    sort_particles_by_cell!,
    spread_forces!,
    stokes_solve!,
    interpolate_velocities!,
    correct_velocities!

# Named problem sizes. The grid doubles per step (three octaves of the
# O(M^3 log M) FFT cost) while N tracks the box volume, holding the particle
# volume fraction at N * (4π/3) / L^3 ≈ 5.1% — semi-dilute, so the
# O(N M_G^3) spread/interpolate cost and the pair-correction cost grow with
# physically consistent loading. Grid spacing h = box/grid = 0.5 keeps
# σ/h ≈ 1.13, the resolution regime of the accuracy tests.
const SIZE_TABLE = (
    small = (box = 16, grid = 32, N = 50),
    medium = (box = 32, grid = 64, N = 400),
    large = (box = 64, grid = 128, N = 3200),
)

# Comma-separated subset of SIZE_TABLE names, e.g. FFCM_BENCHMARK_SIZES=small
# for a quick laptop run. Defaults to all three sizes.
function _selected_sizes()
    names = split(get(ENV, "FFCM_BENCHMARK_SIZES", "small,medium,large"), ",")
    for name in names
        haskey(SIZE_TABLE, Symbol(name)) || throw(ArgumentError(
            "unknown benchmark size $(name); valid sizes: " *
            join(string.(keys(SIZE_TABLE)), ", "),
        ))
    end
    return String.(names)
end

# Keyword arguments for `FFCMConfig{T}` at a named size. Shared by the
# construction benchmarks and `_benchmark_case` so every group measures the
# same physics.
function _config_kwargs(::Type{T}, spec) where {T}
    return (
        L = (T(spec.box), T(spec.box), T(spec.box)),
        R_c = T(2),
        N = spec.N,
        kernel_widths_ratio = T(2),
        num_grid_points = (Int32(spec.grid), Int32(spec.grid), Int32(spec.grid)),
        M_G = 8,
        viscosity = T(1),
    )
end

# Seeded uniform positions in [0, L) and order-one forces. Deterministic
# across runs so judge comparisons measure code, not input variation.
function _random_suspension(::Type{T}, L::NTuple{3, T}, N) where {T}
    rng = Xoshiro(2024)
    Y = Matrix{T}(undef, 3, N)
    F = Matrix{T}(undef, 3, N)
    for n in axes(Y, 2)
        for i in 1:3
            Y[i, n] = L[i] * rand(rng, T)
            F[i, n] = T(2) * rand(rng, T) - T(1)
        end
    end
    return Y, F
end

# One fully warmed case per (T, size): a config whose internal buffers hold a
# valid pipeline state (one `mobility!` call), so each per-step benchmark can
# re-run its step in place. Every step rewrites its outputs from its inputs on
# each call, so repeated timing of a single step is well-posed. Extra keywords
# are forwarded to the constructor.
function _benchmark_case(::Type{T}, spec; config_kwargs...) where {T}
    config = FFCMConfig{T}(; _config_kwargs(T, spec)..., config_kwargs...)
    Y, F = _random_suspension(T, config.L, spec.N)
    V = Matrix{T}(undef, 3, spec.N)
    mobility!(V, config, Y, F)
    return (config = config, Y = Y, F = F, V = V)
end

# Fixed time budget and one evaluation per sample for a hot-path benchmark:
# every leaf runs long enough that BenchmarkTools' tuner would choose
# evals = 1 anyway, and fixed parameters keep runs deterministic.
function _budgeted(benchmark; seconds)
    benchmark.params.seconds = seconds
    benchmark.params.evals = 1
    return benchmark
end

# Opt-in planner-effort experiment, run *separately* from regression runs:
# FFTW consults its in-process wisdom at every planning level, so once a
# measured plan has been built, every later plan of the same transform in the
# same process (including the construction leaves) silently benefits from it.
# Keeping the experiment out of the default suite keeps regression runs
# wisdom-free.
const FFT_PLANNING_EXPERIMENT =
    lowercase(get(ENV, "FFCM_BENCHMARK_FFT_PLANNING", "false")) in ("1", "true")

const SUITE = BenchmarkGroup()

# Backend key: the CPU pipeline today; a future CUDA backend adds sibling
# "cuda" leaves with the same physics so cross-backend speedups come from one
# suite.
cpu = SUITE["cpu"] = BenchmarkGroup()
cpu["construction"] = BenchmarkGroup()
cpu["mobility"] = BenchmarkGroup()
cpu["steps"] = BenchmarkGroup()
cpu["fft-threads"] = BenchmarkGroup()
FFT_PLANNING_EXPERIMENT && (cpu["fft-planning"] = BenchmarkGroup())

for T in (Float32, Float64), name in _selected_sizes()
    spec = SIZE_TABLE[Symbol(name)]
    kwargs = _config_kwargs(T, spec)

    # Cold path. One sample, one evaluation: construction cost (incl. FFT
    # planning) is the metric, and planning beyond the default effort would
    # distort a multi-sample mean anyway.
    cpu["construction"][string(T)][name] = _budgeted(
        (@benchmarkable FFCMConfig{$T}(; $kwargs...) samples = 1); seconds = 60,
    )

    case = _benchmark_case(T, spec)
    config, Y, F, V = case.config, case.Y, case.F, case.V

    # Hot path, assembled operator: the per-iteration cost of a downstream
    # resistance solve, and the headline regression-detector number.
    cpu["mobility"][string(T)][name] = _budgeted(
        (@benchmarkable mobility!($V, $config, $Y, $F)); seconds = 2,
    )

    # Hot path, per step, in pipeline order: localises which cost centre a
    # change moved. Step keys are the step entry-point names.
    steps = cpu["steps"]
    steps["wrap_positions"][string(T)][name] = _budgeted(
        (@benchmarkable wrap_positions!($(config.Y_wrapped), $Y, $(config.L)));
        seconds = 1,
    )
    steps["assign_cells"][string(T)][name] = _budgeted(
        (@benchmarkable assign_cells!($config, $(config.Y_wrapped))); seconds = 1,
    )
    steps["sort_particles_by_cell"][string(T)][name] = _budgeted(
        (@benchmarkable sort_particles_by_cell!($config, $(config.Y_wrapped), $F));
        seconds = 1,
    )
    steps["spread_forces"][string(T)][name] = _budgeted(
        (@benchmarkable spread_forces!($config)); seconds = 1,
    )
    steps["stokes_solve"][string(T)][name] = _budgeted(
        (@benchmarkable stokes_solve!($config)); seconds = 1,
    )
    steps["interpolate_velocities"][string(T)][name] = _budgeted(
        (@benchmarkable interpolate_velocities!($V, $config)); seconds = 1,
    )
    steps["correct_velocities"][string(T)][name] = _budgeted(
        (@benchmarkable correct_velocities!($V, $config)); seconds = 1,
    )

    # Thread scaling of the FFT-based solve: serial plans against plans
    # threaded across whatever `julia -t` provided — never a hardcoded core
    # count. Single-threaded sessions get only the serial leaf.
    cpu["fft-threads"][string(T)][name]["1"] = _budgeted(
        (@benchmarkable stokes_solve!($config)); seconds = 1,
    )
    if Threads.nthreads() > 1
        threaded_case = _benchmark_case(T, spec; fft_threads = Threads.nthreads())
        cpu["fft-threads"][string(T)][name][string(Threads.nthreads())] = _budgeted(
            (@benchmarkable stokes_solve!($(threaded_case.config))); seconds = 1,
        )
    end

    # Transform execution under each planner effort, side by side, on cases
    # pinned explicitly to each effort (independent of the package default).
    if FFT_PLANNING_EXPERIMENT
        estimate_case = _benchmark_case(T, spec; fft_planning = :estimate)
        measured_case = _benchmark_case(T, spec; fft_planning = :measure)
        cpu["fft-planning"]["estimate"][string(T)][name] = _budgeted(
            (@benchmarkable stokes_solve!($(estimate_case.config))); seconds = 1,
        )
        cpu["fft-planning"]["measure"][string(T)][name] = _budgeted(
            (@benchmarkable stokes_solve!($(measured_case.config))); seconds = 1,
        )
    end
end
