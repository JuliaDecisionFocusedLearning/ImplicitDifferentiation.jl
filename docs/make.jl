using ChainRulesCore: ChainRulesCore
using Documenter
using ForwardDiff: ForwardDiff
using ImplicitDifferentiation
using Literate

DocMeta.setdocmeta!(
    ImplicitDifferentiation, :DocTestSetup, :(using ImplicitDifferentiation); recursive=true
)

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
    return title
end

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

example_pages = Pair{String,String}[]
for file in sort(readdir(EXAMPLES_DIR_MD))
    if endswith(file, ".md")
        title = markdown_title(joinpath(EXAMPLES_DIR_MD, file))
        path = joinpath("examples", file)
        push!(example_pages, title => path)
    end
end

pages = [
    "Home" => "index.md",
    "Examples" => example_pages,
    "API reference" => "api.md",
    "FAQ" => "faq.md",
]

fmt = Documenter.HTML(;
    prettyurls=get(ENV, "CI", "false") == "true",
    canonical="https://gdalle.github.io/ImplicitDifferentiation.jl",
    assets=String[],
    edit_link=:commit,
)

makedocs(;
    modules=[ImplicitDifferentiation],
    authors="Guillaume Dalle, Mohamed Tarek and contributors",
    repo="https://github.com/gdalle/ImplicitDifferentiation.jl/blob/{commit}{path}#{line}",
    sitename="ImplicitDifferentiation.jl",
    format=fmt,
    pages=pages,
    linkcheck=true,
)

deploydocs(;
    repo="github.com/gdalle/ImplicitDifferentiation.jl", devbranch="main", push_preview=true
)
