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


def ghums(x, v, N, T,  funcs, epsilons, p_thug=0.8, step_thug=0.1, step_snug=0.01, endpoint=False, adapt_snug=False, snug_target=0.5, δNHUG_min=1e-30, δNHUG_max=100.0, min_pm=1e-2, verbose=True, rng=None):
    """GHUMS Algorithm. Allows both endpoint and non-endpoint versions. This on requires a fixed epsilon schedule."""
    start_time = time.time()
    rng = np.random.default_rng(seed=np.random.randint(low=0, high=10000)) if rng is None else rng
    d = x.shape[1]
    T_size = T if not endpoint else 1  # to allow correct size of arrays for both endpoint and not endpoint versions

    # Storage
    out = {}
    pm_thug = 1.0
    verboseprint = print if verbose else lambda *a, **k: None

    # Initialization (MAKE IT 2D)
    iotas = np.array(rng.binomial(n=1, p=p_thug, size=N))  # 1 == THUG, 0 == SNUG

    try:
        for eps_ix, epsilon in enumerate(epsilons):
            # Check if termination criterion has been reached
            if pm_thug <= min_pm:
                verboseprint("Termination criterion reached.")
                return out
            n = eps_ix + 1
            verboseprint("Iteration: ", n)

            # --- CONSTRUCT TRAJECTORIES ---
            xnk = thug_integrator(x=x, v=v, iotas=iotas, T=T, step_thug=step_thug, step_snug=step_snug)  # (N, T+1, d)
            xnk = xnk[:, [0, -1]] if endpoint else xnk  # either (N, T+1, d) or (N, 2, d)
            verboseprint("\tTrajectories constructed.")

            # --- COMPUTE WEIGHTS ---
            logw = log_post3d(xnk, epsilons[n]) - log_post3d(xnk[:, :1], epsilons[n-1])
            W = np.exp(logw - logsumexp(logw))    # normalized weights, (N, T+1)
            verboseprint("\tWeights computed and normalized.")

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

            # ADAPT STEP SIZE FOR NHUG
            if adapt_snug:
                try:
                    mip_snug = np.quantile(k_indices[iotas[n_indices] == 0], q=0.5) / T_size
                except (ZeroDivisionError, IndexError):
                    verboseprint("\tMIP for SNUG could not be estimated. Setting it to zero.")
                    mip_snug = 0.0
                step_snug = np.clip(np.exp(np.log(step_snug) + 0.5 * (mip_snug - snug_target)), δNHUG_min, δNHUG_max)
                verboseprint("\tSNUG Step size adapted to {:.31f}".format(step_snug))

            # REJUVENATE AUXILIARIES (VELOCITIES + IOTAS)
            v = rng.normal(size=(N, d))
            iotas = rng.binomial(n=1, p=p_thug, size=N)
            verboseprint("\tVelocities and ιs refreshed.")

            n += 1
    except (ValueError, KeyboardInterrupt) as e:
        verboseprint("ValueError was raised: ", e)
        return {
        'εs': εs,
        'total_time': (time() - start_time)
        }
    # Compute quantities before ending the algorithm (we assume I won't be interrupting anything)
    total_time = time() - start_time
    w = exp(logw)
    logsumexp_w = logsumexp(logw)
    EXs_A, EXs_B, ESJDs_A, ESJDs_B = zeros(d), zeros(d), zeros(d), zeros(d)
    ESJDs_A_THUG = zeros(d)
    ESJDs_B_THUG = zeros(d)
    ESJDs_A_NHUG = zeros(d)
    ESJDs_B_NHUG = zeros(d)
    for coord in range(d):
        EXs_A[coord] = estimate_expectation_a(XNK[:, :, coord], w=w)
        EXs_B[coord] = estimate_expectation_b(XNK[:, :, coord], logw=logw)
        ESJDs_A[coord] = compute_esjd_a(XNK[:,:, coord], w=w)
        ESJDs_B[coord] = compute_esjd_b(XNK[:,:, coord], w=w, logsumexp_w=logsumexp_w)
        ESJDs_A_NHUG[coord] = compute_esjd_a(XNK[ι==0,:, coord], w=w[ι==0])
        ESJDs_B_NHUG[coord] = compute_esjd_b(XNK[ι==0,:, coord], w=w[ι==0], logsumexp_w=logsumexp(logw[ι==0]))
        ESJDs_A_THUG[coord] = compute_esjd_a(XNK[ι==1,:, coord], w=w[ι==1])
        ESJDs_B_THUG[coord] = compute_esjd_b(XNK[ι==1,:, coord], w=w[ι==1], logsumexp_w=logsumexp(logw[ι==1]))
    EfXs_A, EfXs_B = zeros(len(funcs)), zeros(len(funcs))
    for fix, f in enumerate(funcs):
        fnk = apply_along_axis(f, 2, XNK)
        EfXs_A[fix] = estimate_expectation_a(fnk, w=w)
        EfXs_B[fix] = estimate_expectation_b(fnk, logw=logw)
    return {
    'εs': εs,
    'EXs_A': EXs_A,
    'EXs_B': EXs_B,
    'ESJDs_A': ESJDs_A,
    'ESJDs_B': ESJDs_B,
    'EfXs_A': EfXs_A,
    'EfXs_B': EfXs_B,
    'ESJDs_A_THUG': ESJDs_A_THUG,
    'ESJDs_B_THUG': ESJDs_B_THUG,
    'ESJDs_A_NHUG': ESJDs_A_NHUG,
    'ESJDs_B_NHUG': ESJDs_B_NHUG,
    'total_time': total_time,
    'n': n,
    'total_time': total_time,
    'PROP_MOVED_THUG': prop_moved_hug
    }