import numpy as np
from ellipsoid_functions import gradient, project, log_post


def mh_kernel(x0, B, step_thug, step_snug, epsilon_target, epsilon_proposal, p_thug, rng=None):
    """Metropolis-Hastings kernel implementing a mixture of HUG and NHUG. This
    is vectorised, meaning it acts on all particles at once. x0 is assumed to be
    a matrix of shape (n_alive_particles, 20). """
    rng = np.random.default_rng(seed=np.random.randint(low=0, high=10000)) if rng is None else rng
    n_alive = x0.shape[0]
    # Sample ι for each particle to determine which kernel to use
    iotas = rng.binomial(n=1, p=p_thug, size=n_alive)
    # Choose step size and sign based on ι
    thug_flag = iotas == 1
    half_step = np.where(thug_flag, 0.5*step_thug, 0.5*step_snug)     # (N_alive,)
    s = np.where(thug_flag, 1, -1)                                    # (N_alive, )
    # Sample all velocities and all log-uniform variables at once
    v0s = rng.normal(loc=0.0, scale=1.0, size=(n_alive, B))            # (N, B)
    logus = np.log(rng.uniform(low=0.0, high=1.0, size=(n_alive, B)))  # (N, B)
    # Acceptance probabilities for each particle
    aps = np.zeros((n_alive, B))
    # Storage for ESJD (Rao-Blackwellised)
    esjd = np.zeros(n_alive)
    # Store ESJD for THUG and NHUG separately
    esjd_thug = np.zeros(np.sum(thug_flag))
    esjd_snug = np.zeros(np.sum(~thug_flag))
    # Loop is still over B since the process is inherently sequential
    for b in range(B):
        # Perform one step of the integrator in a vectorised manner
        x = x0 + half_step[:, np.newaxis]*v0s[:, b, :]
        v = s[:, np.newaxis]*(v0s[:, b, :] - 2*project(v0s[:, b, :], gradient(x)))
        x += half_step[:, np.newaxis]*v
        # Compute acceptance ratios for each particle
        log_ar = log_post(x, epsilon_target) - 0.5*np.linalg.norm(v, axis=1)**2
        log_ar -= log_post(x0, epsilon_proposal) - 0.5*np.linalg.norm(v0s[:, b], axis=1)**2  # (N, )
        ap = np.exp(np.clip(log_ar, a_min=None, a_max=0.0))
        # Compute Rao-Blackwellised ESJD
        esjd += ap * np.linalg.norm(x - x0, axis=1)**2  # (N, )
        # Store ESJD for NHUG and THUG
        esjd_thug += ap[thug_flag] * np.linalg.norm(x[thug_flag] - x0[thug_flag], axis=1)**2
        esjd_snug += ap[~thug_flag] * np.linalg.norm(x[~thug_flag] - x0[~thug_flag], axis=1) ** 2
        # Metropolis-Hastings step
        flag = (logus[:, b] <= log_ar)
        x0[flag] = x[flag]
        aps[flag, b] = 1
    return x0, iotas, aps.mean(axis=1), esjd / B, esjd_snug/B, esjd_thug/B
