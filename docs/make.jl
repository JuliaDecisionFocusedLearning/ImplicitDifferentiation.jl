using ImplicitDifferentiation
using Documenter

DocMeta.setdocmeta!(ImplicitDifferentiation, :DocTestSetup, :(using ImplicitDifferentiation); recursive=true)

makedocs(;
    modules=[ImplicitDifferentiation],
    authors="Guillaume Dalle, Mohamed Tarek and contributors",
    repo="https://github.com/gdalle/ImplicitDifferentiation.jl/blob/{commit}{path}#{line}",
    sitename="ImplicitDifferentiation.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://gdalle.github.io/ImplicitDifferentiation.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/gdalle/ImplicitDifferentiation.jl",
    devbranch="main",
)
