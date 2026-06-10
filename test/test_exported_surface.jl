using Test
using FFCM

# The public surface is the two-phase API (spec/mobility.md): the cold-path
# constructor `FFCMConfig`, the hot-path operator `mobility!`, and the
# matrix-free `FFCMMobility`. The seven internal step functions stay reachable
# as `FFCM.spread_forces!` for tests and advanced use, but exporting them would
# leak the internal decomposition and contradict the deep-module intent.
@testset "Only the two-phase public API is exported" begin
    exported = setdiff(names(FFCM), [:FFCM])   # names() includes the module itself
    @test sort(exported) == sort([:FFCMConfig, :mobility!, :FFCMMobility])
end
