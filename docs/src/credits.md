# Credits and references

PolytopeSamplers.jl is licensed under GPL-3.0-or-later, so that it can adapt code from the
samplers below. Each adapted function names its source, file, commit and license in a
comment.

## Adapted code

| Project | License | Adapted for |
|---|---|---|
| [volesti](https://github.com/GeomScale/volesti) | LGPL-3.0 | maximum-volume ellipsoid, accelerated billiard walk, MMCS rounding phases, sampling tests |
| [PolyRound](https://gitlab.com/csb.ethz/PolyRound) | MIT | ellipsoid rounding safeguards, tests |
| [hops / hopsy](https://github.com/modsim/hopsy) | MIT | coordinate hit-and-run slack updates and cyclic scan, tests |
| [polytopewalk](https://github.com/ethz-randomwalk/polytopewalk) | MIT | Dikin and Vaidya walks |
| [PolytopeSamplerMatlab](https://github.com/ConstrainedSampler/PolytopeSamplerMatlab) | GPL-3.0 | CRHMC integrator and step-size schedule |

## References

- Chalkis, A., Fisikopoulos, V., Tsigaridas, E. & Zafeiropoulos, H. (2021). Geometric
  algorithms for sampling the flux space of metabolic networks. SoCG 2021, 21:1–21:16.
  [doi:10.4230/LIPIcs.SoCG.2021.21](https://doi.org/10.4230/LIPIcs.SoCG.2021.21)
- Chen, Y., Dwivedi, R., Wainwright, M. J. & Yu, B. (2018). Fast MCMC sampling algorithms on
  polytopes. Journal of Machine Learning Research 19, 1–86.
  [jmlr.org/papers/v19/18-158](https://jmlr.org/papers/v19/18-158.html)
- Gryazina, E. & Polyak, B. (2014). Random sampling: billiard walk algorithm. European
  Journal of Operational Research 238, 497–504.
  [doi:10.1016/j.ejor.2014.03.041](https://doi.org/10.1016/j.ejor.2014.03.041)
- Haraldsdóttir, H. S., Cousins, B., Thiele, I., Fleming, R. M. T. & Vempala, S. (2017).
  CHRR: coordinate hit-and-run with rounding for uniform sampling of constraint-based
  models. Bioinformatics 33, 1741–1743.
  [doi:10.1093/bioinformatics/btx052](https://doi.org/10.1093/bioinformatics/btx052)
- Kannan, R. & Narayanan, H. (2012). Random walks on polytopes and an affine interior point
  method for linear programming. Mathematics of Operations Research 37, 1–20.
  [doi:10.1287/moor.1110.0519](https://doi.org/10.1287/moor.1110.0519)
- Kaufman, D. E. & Smith, R. L. (1998). Direction choice for accelerated convergence in
  hit-and-run sampling. Operations Research 46, 84–95.
  [doi:10.1287/opre.46.1.84](https://doi.org/10.1287/opre.46.1.84)
- Kook, Y., Lee, Y. T., Shen, R. & Vempala, S. S. (2022). Sampling with Riemannian
  Hamiltonian Monte Carlo in a constrained space. NeurIPS 35.
  [arXiv:2202.01908](https://arxiv.org/abs/2202.01908)
- Neal, R. M. (2003). Slice sampling. Annals of Statistics 31, 705–767.
  [doi:10.1214/aos/1056562461](https://doi.org/10.1214/aos/1056562461)
- Smith, R. L. (1984). Efficient Monte Carlo procedures for generating points uniformly
  distributed over bounded regions. Operations Research 32, 1296–1308.
  [doi:10.1287/opre.32.6.1296](https://doi.org/10.1287/opre.32.6.1296)
- Vehtari, A., Gelman, A., Simpson, D., Carpenter, B. & Bürkner, P.-C. (2021).
  Rank-normalization, folding, and localization: an improved R̂ for assessing convergence
  of MCMC. Bayesian Analysis 16, 667–718.
  [doi:10.1214/20-BA1221](https://doi.org/10.1214/20-BA1221)
- Zhang, Y. & Gao, L. (2003). On numerical solution of the maximum volume ellipsoid
  problem. SIAM Journal on Optimization 14, 53–76.
  [doi:10.1137/S1052623401397230](https://doi.org/10.1137/S1052623401397230)
