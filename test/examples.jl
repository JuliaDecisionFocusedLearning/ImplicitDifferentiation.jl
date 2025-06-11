@testitem "Intro" begin
    include(joinpath(dirname(@__DIR__), "examples", "0_intro.jl"))
end

@testitem "Basic" begin
    include(joinpath(dirname(@__DIR__), "examples", "1_basic.jl"))
end

@testitem "Advanced" begin
    include(joinpath(dirname(@__DIR__), "examples", "2_advanced.jl"))
end

@testitem "Tricks" begin
    include(joinpath(dirname(@__DIR__), "examples", "3_tricks.jl"))
end
