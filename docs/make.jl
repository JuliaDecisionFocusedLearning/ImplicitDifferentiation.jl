using Documenter
using ForwardDiff: ForwardDiff
using ImplicitDifferentiation
using Literate
using Zygote: Zygote

const ID = ImplicitDifferentiation

DocMeta.setdocmeta!(
    ImplicitDifferentiation, :DocTestSetup, :(using ImplicitDifferentiation); recursive=true
)

cp(
    joinpath(dirname(@__DIR__), "README.md"),
    joinpath(@__DIR__, "src", "index.md");
    force=true,
)

EXAMPLES_DIR_JL = joinpath(dirname(@__DIR__), "examples")
EXAMPLES_DIR_MD = joinpath(@__DIR__, "src", "examples")

for file in readdir(EXAMPLES_DIR_MD)
    if endswith(file, ".md")
        rm(joinpath(EXAMPLES_DIR_MD, file))
    end
end

for file in readdir(EXAMPLES_DIR_JL)
    Literate.markdown(
        joinpath(EXAMPLES_DIR_JL, file),
        EXAMPLES_DIR_MD;
        documenter=true,
        flavor=Literate.DocumenterFlavor(),
    )
end

pages = [
    "Home" => "index.md",
    "Examples" => [
        joinpath("examples", file) for
        file in sort(readdir(EXAMPLES_DIR_MD)) if endswith(file, ".md")
    ],
    "api.md",
    "faq.md",
]

makedocs(;
    modules=[ImplicitDifferentiation],
    authors="Guillaume Dalle, Mohamed Tarek",
    sitename="ImplicitDifferentiation.jl",
    format=Documenter.HTML(),
    pages=pages,
)

deploydocs(;
    repo="github.com/JuliaDecisionFocusedLearning/ImplicitDifferentiation.jl",
    devbranch="main",
)
