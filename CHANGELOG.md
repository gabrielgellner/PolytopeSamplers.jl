# Changelog

All notable changes to this project will be documented in this file.

## [0.1.0] - 2026-11-01

### Breaking Changes

- Rewrite the sampler around Polytope, warmup and hr_sample
- Rename the package to PolytopeSamplers.jl
- Sample through the AbstractMCMC interface and add ESS and R̂ diagnostics
- Round polytopes by maximum-volume ellipsoids and add coordinate hit-and-run (CHRR)
- Default warmup to one linear program, with extreme points computed for ACHR on use
- **sampler**: Report per-step statistics with getstats and store them in Chains internals (#21)

### Bug Fixes

- Bundle samples into a matrix in linear time
- **sampler**: Start chains of sample at random points of the inscribed ball and accept initial_params (#13)

### Documentation

- Write the README and port the constrained food web problem as an example
- Add a parallel food web replicates example
- Show the barrier walks on an unrounded box
- Bring docstrings up to date with the samplers and the default warmup
- Add a Documenter manual built by just docs
- **examples**: Build the food web polytope with FoodWebs.jl (#12)
- **examples**: Keep the pyramid biomasses of the original study in the food web examples (#12)

### Features

- Add the accelerated billiard walk
- Add multiphase Monte Carlo sampling, one-LP warmup and sampling to convergence
- Add the Dikin and Vaidya walks
- Add constrained Riemannian Hamiltonian Monte Carlo (CRHMC)
- Sample non-uniform densities on polytopes with Target
- **sampler**: Return MCMCChains.Chains with chain_type = Chains and add getparams (#14)
- **density**: Add GaussianTarget and ExponentialTarget, drawn exactly on chords by hit-and-run (#20)
- **volume**: Estimate polytope volumes with a standard error by multiphase Monte Carlo with cooling balls (#19)

### Miscellaneous

- Add a justfile, Runic formatting, JETLS lint, CI and a git-cliff changelog
- Leave the pre-conventional history out of the changelog
- Relicense under GPL-3.0-or-later
- Add citation metadata and a TagBot workflow for registered releases (#18)

### Performance

- Update slacks in place and default to a cyclic coordinate scan

### Testing

- Check uniformity against exact and rejection-sampled moments, feasibility, degenerate polytopes, JET and AllocCheck
- Add sampling checks adapted from volesti, hopsy, hops and PolyRound
- Add a convergence stress suite run by just stress
- Check slack updates, the cyclic scan and that samples stay inside the polytope
- Run the Dikin and Vaidya walks through the stress suite
- Run CRHMC through the stress suite
- Check non-uniform targets against exact moments

# Changelog

All notable changes to this project will be documented in this file.
