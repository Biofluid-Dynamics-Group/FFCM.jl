# GPU host↔device staging for the assembled operator (Su & Keaveny 2024, §3-§4). Device
# methods of the staging seam `mobility!` brackets the pipeline with: upload the caller's host
# positions and forces into the device staging buffers before step 1, and copy the device
# velocities back into the caller's host output after step 6. Between the edges the six kernels
# stay device-resident; dispatched on the device `particles` storage type, so `mobility!`
# carries no backend branch (the CPU methods in `src/mobility.jl` are the identity/no-op). A
# caller already holding device arrays is served too — the `copyto!` is then a device→device
# copy. See spec/cuda-conventions.md, "Assembled operator: the host↔device boundary".

function _stage_mobility_io!(
    particles::FFCM.ParticleBuffers{<:CuMatrix, <:CuMatrix}, Y, F, V,
)
    copyto!(particles.Y_input, Y)
    copyto!(particles.F_input, F)
    return (particles.Y_input, particles.F_input, particles.V_output)
end

function _retrieve_mobility_output!(
    V, ::FFCM.ParticleBuffers{<:CuMatrix, <:CuMatrix}, V_staged,
)
    copyto!(V, V_staged)
    return V
end
