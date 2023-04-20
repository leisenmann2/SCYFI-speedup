using SCYFI
using Documenter

DocMeta.setdocmeta!(SCYFI, :DocTestSetup, :(using SCYFI); recursive=true)

makedocs(;
    modules=[SCYFI],
    authors="Lukas Eisenmann",
    repo="https://github.com/Lukas.eisenmann@zi-mannheim.de/SCYFI.jl/blob/{commit}{path}#{line}",
    sitename="SCYFI.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://Lukas.eisenmann@zi-mannheim.de.github.io/SCYFI.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/Lukas.eisenmann@zi-mannheim.de/SCYFI.jl",
    devbranch="main",
)
