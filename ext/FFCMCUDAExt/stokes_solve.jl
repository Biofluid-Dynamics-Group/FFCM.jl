# GPU device kernel for pipeline step 4 — the Fourier-space Stokes solve (Su & Keaveny
# 2024, §3 equations (32)–(33)). `stokes_solve!` is backend-agnostic: its forward/inverse
# transforms go through `mul!` (cuFFT on the device), and the in-place per-mode projection
# `_apply_inverse_stokes_kernel!` gets this device method, dispatched on the device buffer
# types. The kernel mirrors cuFCM's active `cufcm_flow_solve`, monopole (force) only. See
# spec/stokes-solve.md, spec/cuda-conventions.md.

# One thread per Fourier mode over the half-spectrum, grid-stride, 32 threads per block
# (cuFCM's FCM_THREADS_PER_BLOCK). Launch tuning is a deferred, benchmark-gated experiment.
@inline function _stokes_solve_launch(total::Integer)
    threads = 32
    blocks = max(cld(Int(total), threads), 1)
    return threads, blocks
end

# Each thread takes one Fourier mode of the half-spectrum `(M_x÷2+1, M_y, M_z)`: decompose
# its column-major linear index into `(ix, iy, iz)`, read the device-resident wavenumbers,
# and apply the incompressible projection scaled by the Stokeslet `1/(μ k²)` with the FFT
# round-trip's `1/M` folded into `inv_μ_M`. The projection is local per mode, so the in-place
# `fluid_hat` update is race-free; the k = 0 mode is a guarded write of zero (gauge fix).
function _apply_inverse_stokes_device!(
    fx̂, fŷ, fẑ,
    k_x, k_y, k_z,
    inv_μ_M::T,
) where {T}
    fft_M_x = size(fx̂, 1)
    M_y = size(fx̂, 2)
    plane = fft_M_x * M_y
    total = length(fx̂)

    index = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x
    for lin in index:stride:total
        lin0 = lin - 1
        iz = lin0 ÷ plane + 1
        rest = lin0 - (iz - 1) * plane
        iy = rest ÷ fft_M_x + 1
        ix = rest - (iy - 1) * fft_M_x + 1
        @inbounds begin
            k = SVector(k_x[ix], k_y[iy], k_z[iz])
            k² = dot(k, k)
            if iszero(k²)
                f̂ = zero(SVector{3, Complex{T}})
            else
                f̂ = SVector(fx̂[lin], fŷ[lin], fẑ[lin])
                inv_k² = one(T) / k²
                f̂ = (inv_μ_M * inv_k²) * (f̂ - k * (dot(k, f̂) * inv_k²))
            end
            fx̂[lin] = f̂[1]
            fŷ[lin] = f̂[2]
            fẑ[lin] = f̂[3]
        end
    end
    return nothing
end

function _apply_inverse_stokes_kernel!(
    fx̂::CuArray{Complex{T}, 3},
    fŷ::CuArray{Complex{T}, 3},
    fẑ::CuArray{Complex{T}, 3},
    k_x::CuVector{T},
    k_y::CuVector{T},
    k_z::CuVector{T},
    μ::T,
    inv_M::T,
) where {T}
    inv_μ_M = inv_M / μ
    threads, blocks = _stokes_solve_launch(length(fx̂))
    @cuda threads = threads blocks = blocks _apply_inverse_stokes_device!(
        fx̂, fŷ, fẑ,
        k_x, k_y, k_z,
        inv_μ_M,
    )
    return nothing
end
