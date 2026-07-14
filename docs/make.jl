using Documenter
using FFCM

# `checkdocs = :exports`: the internal step functions carry docstrings but are
# deliberately not part of the documented surface, so only exported names are
# required on the pages. `remotes = nothing` and the `nothing` HTML link
# settings: the build is local-only for now, so all links to a remote
# repository (source, edit, navbar) are disabled.
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
    remotes = nothing,
    format = Documenter.HTML(;
        prettyurls = get(ENV, "CI", nothing) == "true",
        edit_link = nothing,
        repolink = nothing,
    ),
)
