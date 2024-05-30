import time
import numpy as np
from scipy.special import logsumexp
from ellipsoid_functions import gradient, project, log_post
from ghums_functions import thug_integrator


def mh_kernel_product(x0, T, step_thug, step_snug, epsilon_target, p_thug, rng=None):
    """Metropolis-Hastings kernel implementing a mixture of HUG and NHUG. It performs T MH steps, hence the name
    `product`. This is vectorised, meaning it acts on all particles at once. x0 is assumed to be
    a matrix of shape (n_alive_particles, 20). """
    rng = np.random.default_rng(seed=np.random.randint(low=0, high=10000)) if rng is None else rng
    N, d = x0.shape
    # Sample ι for each particle to determine which kernel to use
    iotas = rng.binomial(n=1, p=p_thug, size=N)
    # Choose step size and sign based on ι
    thug_flag = iotas == 1
    half_step = np.where(thug_flag, 0.5*step_thug, 0.5*step_snug)     # (N,)
    s = np.where(thug_flag, 1, -1)                                    # (N, )
    # Sample all velocities and all log-uniform variables at once
    v0s = rng.normal(loc=0.0, scale=1.0, size=(N, T, d))            # (N, B)
    logus = np.log(rng.uniform(low=0.0, high=1.0, size=(N, T)))  # (N, B)
    # Acceptance probabilities for each particle
    aps = np.zeros((N, T))
    # Storage for ESJD (Rao-Blackwellised)
    esjd = np.zeros(N)
    # Store ESJD for THUG and NHUG separately
    esjd_thug = np.zeros(np.sum(thug_flag))
    esjd_snug = np.zeros(np.sum(~thug_flag))
    # Loop is still over B since the process is inherently sequential
    for k in range(T):
        # Perform one step of the integrator in a vectorised manner
        x = x0 + half_step[:, np.newaxis]*v0s[:, k, :]
        v = s[:, np.newaxis]*(v0s[:, k, :] - 2*project(v0s[:, k, :], gradient(x)))
        x = x + half_step[:, np.newaxis]*v
        # Compute acceptance ratios for each particle
        log_ar = log_post(x, epsilon_target) - 0.5*np.linalg.norm(v, axis=1)**2
        log_ar -= log_post(x0, epsilon_target) - 0.5*np.linalg.norm(v0s[:, k], axis=1)**2  # (N, )
        ap = np.exp(np.clip(log_ar, a_min=None, a_max=0.0))
        # Compute Rao-Blackwellised ESJD
        esjd += ap * np.linalg.norm(x - x0, axis=1)**2  # (N, )
        # Store ESJD for NHUG and THUG
        esjd_thug += ap[thug_flag] * np.linalg.norm(x[thug_flag] - x0[thug_flag], axis=1)**2
        esjd_snug += ap[~thug_flag] * np.linalg.norm(x[~thug_flag] - x0[~thug_flag], axis=1) ** 2
        # Metropolis-Hastings step
        flag = (logus[:, k] <= log_ar)
        x0[flag] = x[flag]
        aps[flag, k] = 1
    return x0, iotas, aps.mean(axis=1), esjd / T, esjd_snug/T, esjd_thug/T


def mh_kernel_endpoint(x0, T, step_thug, step_snug, epsilon_target, p_thug, rng=None):
    """Constructs an entire trajectory and then does a single MH step at the end."""
    rng = np.random.default_rng(seed=np.random.randint(low=0, high=10000)) if rng is None else rng
    N, d = x0.shape
    # Sample ι for each particle to determine which kernel to use
    iotas = rng.binomial(n=1, p=p_thug, size=N)
    # Choose step size and sign based on ι
    thug_flag = iotas == 1
    # Sample one velocity for each particle
    v0s = rng.normal(loc=0.0, scale=1.0, size=(N, d))            # (N, d)
    logus = np.log(rng.uniform(low=0.0, high=1.0, size=N))       # (N, )
    # Acceptance probabilities for each particle
    aps = np.zeros(N)
    # Construct trajectories
    x = thug_integrator(x=x0, v=v0s, iotas=iotas, T=T, step_thug=step_thug, step_snug=step_snug)[:, -1]
    # Compute acceptance ratios for each particle
    log_ar = log_post(x, epsilon_target)  # - 0.5 * np.linalg.norm(v, axis=1) ** 2
    log_ar -= log_post(x0, epsilon_target)  # - 0.5 * np.linalg.norm(v0s, axis=1) ** 2  # (N, )
    ap = np.exp(np.clip(log_ar, a_min=None, a_max=0.0))
    # Compute ESJDs
    esjd = np.sum(ap * np.linalg.norm(x - x0, axis=1) ** 2)  # (N, )
    esjd_thug = np.sum(ap[thug_flag] * np.linalg.norm(x[thug_flag] - x0[thug_flag], axis=1) ** 2)
    esjd_snug = np.sum(ap[~thug_flag] * np.linalg.norm(x[~thug_flag] - x0[~thug_flag], axis=1) ** 2)
    # MH
    flag = (logus <= log_ar)
    x0[flag] = x[flag]
    aps[flag] = 1
    return x0, iotas, aps.mean(axis=1), esjd, esjd_snug, esjd_thug


def smc(x, epsilons, N=5000, T=10, step_thug=0.01, step_snug=0.01, p_thug=0.5, snug_min_step=1e-30, snug_max_step=100.0,
        snug_target=0.5, min_ap=1e-2, verbose=True, rng=None, mode='product'):
    """Implements an SMC sampler with mixture of HUG and NHUG kernels."""
    assert mode in {'endpoint', 'product'}, "Mode must be either endpoint or product."
    M_kernel = mh_kernel_endpoint if mode == 'endpoint' else mh_kernel_product
    start_time = time.time()
    rng = np.random.default_rng(seed=np.random.randint(low=0, high=10000)) if rng is None else rng
    W = np.full(N, 1/N)
    out = {'ess': [], 'thug_ap': [1.0], 'runtime': 0.0, 'epsilons': epsilons, 'esjd': []}
    verboseprint = print if verbose else lambda *a, **k: None

    n = 0
    try:
        while out['thug_ap'][-1] > min_ap and n < len(epsilons)-1:
            verboseprint("Iteration: ", n, " Epsilon: ", epsilons[n])

            # --- RESAMPLING ---
            indices = rng.choice(a=N, size=N, p=W, replace=True)
            x = x[indices]
            verboseprint("\tParticles resampled.")

            # --- MUTATE ---
            x, iotas, aps, esjd, esjd_thug_alive, esjd_snug_alive = M_kernel(
                x0=x, T=T, step_thug=step_thug, step_snug=step_snug, epsilon_target=epsilons[n],
                p_thug=p_thug, rng=rng)
            out['esjd'].append(esjd.mean())
            verboseprint("\tParticles Mutated.")
            verboseprint("\tESJD: ", esjd.mean())

            # --- ESTIMATE ACCEPTANCE PROBABILITY ---
            thug_ap = aps[iotas == 1].mean()
            snug_ap = aps[iotas == 0].mean()
            out['thug_ap'].append(thug_ap)
            verboseprint("\tTHUG AP:  {:.16f}".format(thug_ap))
            verboseprint("\tNHUG AP:  {:.16f}".format(snug_ap))

            # --- ADAPT NHUG STEP SIZE ---
            step_snug = np.clip(np.exp(np.log(step_snug) + 0.5*(snug_ap - snug_target)), snug_min_step, snug_max_step)
            verboseprint("\tSNUG Step size adapted to {:.31f}".format(step_snug))

            # --- RE-WEIGHTING ---
            logw = log_post(x, epsilons[n+1]) - log_post(x, epsilons[n])
            W = np.exp(logw - logsumexp(logw))
            out['ess'].append(1 / np.sum(W**2))
            verboseprint("\tWeights computed and normalized. ESS: ", out['ess'][-1])

            n += 1
    except (ValueError, KeyboardInterrupt) as e:
        verboseprint("Error was raised: ", e)
        out['runtime'] = time.time() - start_time
        return out
    out['runtime'] = time.time() - start_time
    return out
