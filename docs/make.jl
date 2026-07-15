using Documenter
using FFCM

# `checkdocs = :exports`: the internal step functions carry docstrings surfaced
# on the developer pages, but only the exported names are required to appear.
makedocs(;
    sitename = "FFCM.jl",
    modules = [FFCM],
    pages = [
        "Home" => "index.md",
        "Method" => [
            "The mobility operator" => "method/overview.md",
            "Cell lists" => "method/cell-lists.md",
            "Spreading and interpolation" => "method/spreading-interpolation.md",
            "The spectral Stokes solve" => "method/stokes-solve.md",
            "The pairwise correction" => "method/pairwise-correction.md",
        ],
        "GPU acceleration" => "gpu.md",
        "Performance tips" => "performance.md",
        "Notation" => "notation.md",
        "API reference" => "api.md",
        "Developer documentation" => [
            "Architecture" => "devdocs/architecture.md",
            "GPU architecture" => "devdocs/gpu-architecture.md",
        ],
    ],
    checkdocs = :exports,
    format = Documenter.HTML(;
        prettyurls = get(ENV, "CI", nothing) == "true",
        canonical = "https://biofluid-dynamics-group.github.io/FFCM.jl",
        edit_link = "main",
    ),
)

deploydocs(;
    repo = "github.com/Biofluid-Dynamics-Group/FFCM.jl",
    devbranch = "main",
    push_preview = false,
)
