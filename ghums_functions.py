import time
import numpy as np
from scipy.special import logsumexp
from ellipsoid_functions import gradient, project, log_post3d


def thug_integrator(x, v, iotas, T, step_thug, step_snug):
    """Vectorised THUG integrator. Performs both THUG and NHUG at once, using `iotas` to decide which one.

    Parameters
    ----------
    :param x: Seed positions (N, d)
    :type x: np.ndarray
    :param v: Seed velocities (N, d)
    :type v: np.ndarray
    :param iotas: Array of indicators for THUG (iotas == 1) or SNUG (iotas == 0)
    :type iotas: np.ndarray
    :param T: Number of integration steps
    :type T: int
    :param step_thug: Step size for THUG
    :type step_thug: float
    :param step_snug: Step size for SNUG
    :type step_snug: float
    :return xnk: Position trajectories for each particle (N, B+1, d)
    :rtype: np.ndarray
    """
    N = x.shape[0]
    d = x.shape[1]
    # Choose step size and sign based on ι
    half_step = np.where(iotas == 1, 0.5*step_thug, 0.5*step_snug)  # (N,)
    s = np.where(iotas == 1, 1, -1)                                 # (N,)
    # Here we need to generate the full trajectory, so we initialize the array
    xnk = np.zeros((N, T+1, d))
    xnk[:, 0] = x
    for k in range(T):
        # Move to midpoint
        xnk[:, k+1] = xnk[:, k] + half_step[:, np.newaxis]*v            # midpoint is (N, 20)
        # Bounce velocities
        v = s[:, np.newaxis]*(v - 2*project(v, gradient(xnk[:, k+1])))  # (N, 20)
        # Move to final point
        xnk[:, k+1] += half_step[:, np.newaxis]*v
    return xnk


def esjd_a(fnk, w):
    """Computes ESJD of type A."""
    T = fnk.shape[1]-1
    ESJD = 0.0
    w_denominator = w.sum(axis=0)
    for k_ix in range(T+1):
        for l_ix in range(k_ix+1, T+1):
            wl = w[:, l_ix]/w_denominator[l_ix] if w_denominator[l_ix] > 0 else 0.0
            wk = w[:, k_ix]/w_denominator[k_ix] if w_denominator[k_ix] > 0 else 0.0
            ESJD += np.sum((fnk[:, l_ix]*wl - fnk[:, k_ix]*wk)**2)
    return ESJD / ((T+1)**2)


def esjd_b(fnk, w, lse_w):
    """Computes ESJD of type B."""
    fnk_weighted = fnk * w
    # Utilize broadcasting and vectorization for computations
    differences = fnk_weighted[:, None, :] - fnk_weighted[:, :, None]
    squared_differences = np.sum(differences ** 2, axis=0)
    # Efficient summation across the required range
    ESJD = np.triu(squared_differences, k=1).sum()
    return ESJD / np.exp(2 * lse_w)


def ghums(x, v, N, T,  funcs, epsilons, p_thug=0.8, step_thug=0.1, step_snug=0.01, endpoint=False, adapt_snug=False,
          snug_target=0.5, snug_min_step=1e-30, snug_max_step=100.0, min_pm=1e-2, verbose=True, rng=None):
    """GHUMS Algorithm. Allows both endpoint and non-endpoint versions. This on requires a fixed epsilon schedule."""
    start_time = time.time()
    rng = np.random.default_rng(seed=np.random.randint(low=0, high=10000)) if rng is None else rng
    d = x.shape[1]
    nf = len(funcs)
    T_size = T if not endpoint else 1  # to allow correct size of arrays for both endpoint and not endpoint versions
    logw = np.full(N*(T+1), np.nan)
    xnk = np.full(shape=(N, T_size+1, d), fill_value=np.nan)

    # Storage
    out = {'pm_thug': [1.0], 'mip_snug': [], 'snug_steps': [step_snug], 'runtime': 0.0, 'folded_ess': [],
           'epsilons': epsilons}
    verboseprint = print if verbose else lambda *a, **k: None

    # Initialization (MAKE IT 2D)
    iotas = np.array(rng.binomial(n=1, p=p_thug, size=N))  # 1 == THUG, 0 == SNUG

    n = 0
    try:
        while out['pm_thug'][-1] > min_pm and n < len(epsilons)-1:
            verboseprint("Iteration: ", n, " Epsilon: ", epsilons[n])

            # --- CONSTRUCT TRAJECTORIES ---
            xnk = thug_integrator(x=x, v=v, iotas=iotas, T=T, step_thug=step_thug, step_snug=step_snug)  # (N, T+1, d)
            xnk = xnk[:, [0, -1]] if endpoint else xnk  # either (N, T+1, d) or (N, 2, d)
            verboseprint("\tTrajectories constructed.")

            # --- COMPUTE WEIGHTS ---
            logw = log_post3d(xnk, epsilons[n+1]) - log_post3d(xnk[:, :1], epsilons[n])
            W = np.exp(logw - logsumexp(logw))    # normalized weights, (N, T+1)
            logw_folded = logsumexp(logw, axis=1) - np.log(T+1)  # folded weights
            W_folded = np.exp(logw_folded - logsumexp(logw_folded))
            out['folded_ess'].append(1 / np.sum(W_folded**2))
            verboseprint("\tWeights computed and normalized. Folded ESS {:.3f}".format(out['folded_ess'][-1]))

            # --- RESAMPLING ---
            indices = rng.choice(a=N*(T+1), size=N, replace=True, p=W.ravel())   # (N, )
            n_indices, k_indices = np.unravel_index(indices, (N, T_size+1))  # (N, ) and (N, )
            x = xnk[n_indices, k_indices]
            verboseprint("\tParticles Resampled.")

            # Estimate the proportion of particles moved for THUG. Used as termination criterion
            try:
                k_indices_thug = k_indices[iotas[n_indices] == 1]
                pm_thug = sum(k_indices_thug >= 1) / len(k_indices_thug)
            except (ZeroDivisionError, IndexError):
                verboseprint("\tPM for THUG could not be estimated. Setting it to zero.")
                pm_thug = 0.0
            out['pm_thug'].append(pm_thug)

            # ADAPT STEP SIZE FOR NHUG
            if adapt_snug:
                try:
                    mip_snug = np.quantile(k_indices[iotas[n_indices] == 0], q=0.5) / T_size
                except (ZeroDivisionError, IndexError):
                    verboseprint("\tMIP for SNUG could not be estimated. Setting it to zero.")
                    mip_snug = 0.0
                out['mip_snug'].append(mip_snug)
                step_snug = np.clip(
                    np.exp(np.log(step_snug) + 0.5 * (mip_snug - snug_target)), snug_min_step, snug_max_step
                )
                verboseprint("\tSNUG Step size adapted to {:.31f}".format(step_snug))
                out['snug_steps'].append(step_snug)

            # REJUVENATE AUXILIARIES (VELOCITIES + IOTAS)
            v = rng.normal(size=(N, d))
            iotas = rng.binomial(n=1, p=p_thug, size=N)
            verboseprint("\tVelocities and ιs refreshed.")

            n += 1
    except (ValueError, KeyboardInterrupt) as e:
        verboseprint("ValueError was raised: ", e)
        out['runtime'] = time.time() - start_time
        return out
    out['runtime'] = time.time() - start_time
    # For this repo we only need to compute ESJD-A, ESJD-B for the test functions
    w = np.exp(logw)         # un-normalized log weights (N, T+1)
    lse_w = logsumexp(logw)  # scalar, log-sum-exp of un-normalized log weights
    out['ESJD-A'], out['ESJD-B'], out['ESJD-A-THUG'] = np.zeros(nf), np.zeros(nf), np.zeros(nf)
    out['ESJD-B-THUG'], out['ESJD-A-SNUG'], out['ESJD-B-SNUG'] = np.zeros(nf), np.zeros(nf), np.zeros(nf)
    for fix, f in enumerate(funcs):
        fnk = np.apply_along_axis(f, 2, xnk)
        out['ESJD-A'][fix] = esjd_a(fnk, w)
        out['ESJD-B'][fix] = esjd_b(fnk, w, lse_w)
        out['ESJD-A-THUG'][fix] = esjd_a(fnk[iotas == 1], w[iotas == 1])
        out['ESJD-B-THUG'][fix] = esjd_b(fnk[iotas == 1], w[iotas == 1], lse_w=logsumexp(logw[iotas == 1]))
        out['ESJD-A-SNUG'][fix] = esjd_a(fnk[iotas == 0], w[iotas == 0])
        out['ESJD-B-SNUG'][fix] = esjd_b(fnk[iotas == 0], w[iotas == 0], lse_w=logsumexp(logw[iotas == 0]))
    verboseprint("Final epsilon: ", epsilons[n], " ESJD-A: ", out['ESJD-A'][0])
    return out
