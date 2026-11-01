# API reference

```@meta
CurrentModule = PolytopeSamplers
```

## Polytopes and warmup

```@docs
Polytope
isfeasible
Base.isempty(::Polytope)
warmup
Warmup
```

## Targets

```@docs
Target
```

## Sampling

```@docs
sample(::AbstractRNG, ::Polytope, ::PolytopeSampler, ::Integer)
sample_until_converged
mmcs
diagnostics
```

## Samplers

```@docs
BilliardWalk
CoordinateHitAndRun
HitAndRun
CRHMC
DikinWalk
VaidyaWalk
ACHR
PolytopeSampler
AbstractHitAndRun
```

`MCMCThreads`, `MCMCSerial` and `MCMCDistributed` are re-exported from AbstractMCMC.
