using Test
using Aqua
using ForceCouplingMethod

@testset "Aqua: package hygiene" begin
    Aqua.test_all(ForceCouplingMethod)
end
