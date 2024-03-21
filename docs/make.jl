using ChainRulesCore: ChainRulesCore
using Documenter
using ForwardDiff: ForwardDiff
using ImplicitDifferentiation
using Literate
using Zygote: Zygote

const ID = ImplicitDifferentiation

DocMeta.setdocmeta!(
    ImplicitDifferentiation, :DocTestSetup, :(using ImplicitDifferentiation); recursive=true
)

base_url = "https://github.com/gdalle/ImplicitDifferentiation.jl/blob/main/"

open(joinpath(@__DIR__, "src", "index.md"), "w") do io
    # Point to source license file
    println(
        io,
        """
        ```@meta
        EditURL = "$(base_url)README.md"
        ```
        """,
    )
    # Write the contents out below the meta block
    for line in eachline(joinpath(dirname(@__DIR__), "README.md"))
        println(io, line)
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
    "FAQ" => "faq.md",
    "Examples" => example_pages,
    "API reference" => "api.md",
]

fmt = Documenter.HTML(;
    prettyurls=get(ENV, "CI", "false") == "true",
    canonical="https://gdalle.github.io/ImplicitDifferentiation.jl",
    assets=String[],
    edit_link=:commit,
)

if isdefined(Base, :get_extension)
    extension_modules = [
        Base.get_extension(ID, :ImplicitDifferentiationChainRulesCoreExt),
        Base.get_extension(ID, :ImplicitDifferentiationForwardDiffExt),
    ]
else
    extension_modules = [
        ID.ImplicitDifferentiationChainRulesCoreExt,
        ID.ImplicitDifferentiationForwardDiffExt,
    ]
end

makedocs(;
    modules=[ImplicitDifferentiation],
    authors="Guillaume Dalle, Mohamed Tarek and contributors",
    repo=Documenter.Remotes.GitHub("gdalle", "ImplicitDifferentiation.jl"),
    sitename="ImplicitDifferentiation.jl",
    format=fmt,
    pages=pages,
    linkcheck=true,
)

deploydocs(;
    repo="github.com/gdalle/ImplicitDifferentiation.jl", devbranch="main", push_preview=true
)

rm(joinpath(@__DIR__, "src", "index.md"))
