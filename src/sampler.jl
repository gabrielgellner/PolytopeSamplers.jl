"""Sample a new feasible point from the point `x` in direction `delta`."""
function step(sampler, x, delta, fraction = nothing, tries = 0)
    prob = sampler.problem #::HRProblem
    valid = ((abs(delta) > feasibility_tol) & np.logical_not(prob.variable_fixed))
    # permissible alphas for staying in variable bounds
    valphas = ((1.0 - bounds_tol) * prob.variable_bounds - x)[:, valid]
    valphas = (valphas / delta[valid]).flatten()
    if prob.bounds.shape[1] > 0
        # permissible alphas for staying in constraint bounds
        ineqs = prob.inequalities.dot(delta)
        valid = abs(ineqs) > feasibility_tol
        balphas = ((1.0 - bounds_tol) * prob.bounds -
                   prob.inequalities.dot(x))[:, valid]
        balphas = (balphas / ineqs[valid]).flatten()
        # combined alphas
        alphas = np.hstack([valphas, balphas])
    else
        alphas = valphas
    end
    pos_alphas = alphas[alphas > 0.0]
    neg_alphas = alphas[alphas <= 0.0]
    alpha_range = np.array([neg_alphas.max() if len(neg_alphas) > 0 else 0,
                            pos_alphas.min() if len(pos_alphas) > 0 else 0])

    if fraction
        alpha = alpha_range[1] + fraction * (alpha_range[2] - alpha_range[1])
    else:
        alpha = np.random.uniform(alpha_range[1], alpha_range[2])
    end
    p = x + alpha * delta

    # Numerical instabilities may cause bounds invalidation
    # reset sampler and sample from one of the original warmup directions
    # if that occurs. Also reset if we got stuck.
    if (np.any(sampler._bounds_dist(p) < -bounds_tol) || np.abs(np.abs(alpha_range).max() * delta).max() < bounds_tol)
        if tries > max_tries
            raise RuntimeError("Can not escape sampling region, model seems"
                               " numerically unstable :( Reporting the "
                               "model to "
                               "https://github.com/opencobra/cobrapy/issues "
                               "will help us to fix this :)")
        end
        LOGGER.info("found bounds infeasibility in sample, resetting to center")
        newdir = sampler.warmup[np.random.randint(sampler.n_warmup)]
        sampler.retries += 1
        return step(sampler, sampler.center, newdir - sampler.center, nothing, tries + 1)
    end
    return p
end

# ACHR
function ACHR_step()
    pi = np.random.randint(self.n_warmup)
    # mix in the original warmup points to not get stuck
    delta = self.warmup[pi, ] - self.center
    self.prev = step(self, self.prev, delta)
    if self.problem.homogeneous && (self.n_samples * self.thinning % self.nproj == 0)
        self.prev = self._reproject(self.prev)
        self.center = self._reproject(self.center)
    end
    self.center = ((self.n_samples * self.center) / (self.n_samples + 1) +
                   self.prev / (self.n_samples + 1))
    self.n_samples += 1
end
