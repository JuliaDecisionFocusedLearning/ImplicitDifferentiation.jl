using TestItems

@testitem "Settings" begin
    @test startswith(string(ImplicitFunction(nothing, nothing)), "ImplicitFunction")
end
