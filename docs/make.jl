using Documenter
using ImplicitDifferentiation
using Literate

DocMeta.setdocmeta!(
    ImplicitDifferentiation, :DocTestSetup, :(using ImplicitDifferentiation); recursive=true
)

EXAMPLES_DIR_JL = joinpath(dirname(@__DIR__), "test")
EXAMPLES_DIR_MD = joinpath(@__DIR__, "src", "examples")

for file in readdir(EXAMPLES_DIR_MD)
    if endswith(file, ".md")
        rm(joinpath(EXAMPLES_DIR_MD, file))
    end
end

for file in readdir(EXAMPLES_DIR_JL)
    if (!endswith(file, ".jl")) ||
        startswith(file, "runtests") ||
        startswith(file, "profiling")
        continue
    else
        Literate.markdown(
            joinpath(EXAMPLES_DIR_JL, file),
            EXAMPLES_DIR_MD;
            documenter=true,
            flavor=Literate.DocumenterFlavor(),
        )
    end
end

function markdown_title(path)
    title = "?"
    open(path, "r") do file
        for line in eachline(file)
            if startswith(line, '#')
                title = strip(line, [' ', '#'])
                break
            end
        end
    end
    return String(title)
end

pages = [
    "Home" => "index.md",
    "Mathematical background" => "background.md",
    "API reference" => "api.md",
    "Examples" => [
        markdown_title(joinpath(EXAMPLES_DIR_MD, file)) => joinpath("examples", file)
        for file in sort(readdir(EXAMPLES_DIR_MD)) if endswith(file, ".md")
    ],
]

makedocs(;
    modules=[ImplicitDifferentiation],
    authors="Guillaume Dalle, Mohamed Tarek and contributors",
    repo="https://github.com/gdalle/ImplicitDifferentiation.jl/blob/{commit}{path}#{line}",
    sitename="ImplicitDifferentiation.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://gdalle.github.io/ImplicitDifferentiation.jl",
        assets=String[],
        edit_link=:commit,
    ),
    pages=pages,
)

deploydocs(; repo="github.com/gdalle/ImplicitDifferentiation.jl", devbranch="main")
