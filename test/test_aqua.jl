using Test
using Aqua
using FFCM

@testset "Aqua: package hygiene" begin
    Aqua.test_all(FFCM)
end
