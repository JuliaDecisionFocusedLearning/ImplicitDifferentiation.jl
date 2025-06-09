@testitem "intro" begin
    include(joinpath(dirname(@__DIR__), "examples", "0_intro.jl"))
end

@testitem "basic" begin
    include(joinpath(dirname(@__DIR__), "examples", "1_basic.jl"))
end

@testitem "advanced" begin
    include(joinpath(dirname(@__DIR__), "examples", "2_advanced.jl"))
end

@testitem "tricks" begin
    include(joinpath(dirname(@__DIR__), "examples", "3_tricks.jl"))
end
