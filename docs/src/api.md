# API reference

The public surface is three names: a configuration constructor (the cold path, run
once), the mutating operator application (the hot path, run per solver iteration), and
a matrix-free operator wrapper for iterative solvers. The
[method overview](method/overview.md) explains how they fit together; internal step
functions are documented in the [architecture](devdocs/architecture.md) page of the
developer documentation.

```@docs
FFCMConfig
mobility!
FFCMMobility
```
