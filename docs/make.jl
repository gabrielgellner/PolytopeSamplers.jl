using Documenter
using PolytopeSamplers

DocMeta.setdocmeta!(
    PolytopeSamplers, :DocTestSetup, :(using PolytopeSamplers, Random); recursive = true,
)

makedocs(;
    modules = [PolytopeSamplers],
    sitename = "PolytopeSamplers.jl",
    authors = "Gabriel Gellner",
    format = Documenter.HTML(;
        prettyurls = get(ENV, "CI", nothing) == "true",
        canonical = "https://gabrielgellner.github.io/PolytopeSamplers.jl",
        edit_link = "main",
    ),
    pages = [
        "Home" => "index.md",
        "Guide" => [
            "Defining polytopes and warmup" => "polytopes.md",
            "Sampling and convergence" => "sampling.md",
            "Choosing a sampler" => "samplers.md",
            "Non-uniform targets" => "targets.md",
            "Parallel sampling" => "parallel.md",
        ],
        "Example: constrained food webs" => "foodwebs.md",
        "Background" => [
            "How the samplers are tested" => "testing.md",
            "Credits and references" => "credits.md",
        ],
        "API reference" => "api.md",
    ],
    checkdocs = :exports,
)

# Deploys only on GitHub Actions; a local build stops here
deploydocs(; repo = "github.com/gabrielgellner/PolytopeSamplers.jl", devbranch = "main")
