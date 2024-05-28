import numpy as np


def level_set(d):
    """Computes a good level set for a dimension."""
    return np.ceil(0.5*(d//2)*np.log(10) - (d//2)*np.log(2*np.pi) - 0.5*(1.1*(d//2)))


def constraint(x):
    """Constraint function representing an ellipse. This ellipse corresponds to the `level_set(d)`-level set of a
    d-dimensional normal distribution with covariance matrix `Sigma = np.diag([1, 0.1]*(d//2))`. The constraint function
    will return 0.0 when the point `x` is on the ellipse.

    Parameters
    ----------
    :param x: Vector at which we wish to compute the constraint function at
    :type x: np.ndarray
    :return: Constraint function evaluated at x
    :rtype: float
    """
    d = x.shape[1]
    return (d/4)*np.log(10) - (d//2)*np.log(2*np.pi) - 0.5*np.multiply(
        np.tile(np.array([1., 10.]), (len(x), d//2)), x**2).sum(axis=1) - level_set(d)


def gradient(x):
    """Computes the gradient of the constraint function for each row of x.

    Parameters
    ----------
    :param x: (N, d) Matrix of inputs at which we wish to compute the gradient of the constraint function
    :type x: np.ndarray
    :return: (N, d) Matrix of gradients of the constraint function at each row of x
    :rtype: np.ndarray
    """
    return -np.multiply(np.tile(np.array([1., 10.]), (len(x), x.shape[1]//2)), x)


def log_post(x, epsilon):
    """Log posterior density with respect to the Lebesgue measure. This is the product of a d-dimensional standard
    normal and an indicator function that checks if `constraint(x)` is within a tolerance of `epsilon`.

    Parameters
    ----------
    :param x: Vector at which to evaluate log posterior (N, d)
    :type x: np.ndarray
    :param epsilon: Tolerance for the indicator function
    :type epsilon: float
    :return: Log posterior densities at each row of x, resulting in a (N, ) array
    :rtype: float
    """
    d = x.shape[1]
    log_density_values = np.full(len(x), -np.inf)
    flag = np.abs(constraint(x)) <= epsilon
    log_density_values[flag] = -0.5*(np.linalg.norm(x[flag], axis=1)**2) - (d//2)*np.log(2*np.pi)
    return log_density_values


def project(vs, gs):
    """Projects every row in `vs` using the corresponding row in `gs`.

    Parameters
    ----------
    :param vs: Matrix of velocities for THUG to be projected using `gs`
    :type vs: np.ndarray
    :param gs: Matrix of gradients at the midpoints. Array `gs[i]` used to project `vs[i]`.
    :type gs: np.ndarray
    :return: Projected velocities in a (N, d) array
    :rtype: np.ndarray
    """
    gs_hat = gs / np.linalg.norm(gs, axis=1, keepdims=True)  # normalize every row of g (N, 20)
    return np.multiply(gs_hat, np.einsum('ij,ij->i', vs, gs_hat)[:, None])


def thug_integrator(x, v, iotas, B, step_thug, step_snug):
    """Vectorised THUG integrator. Performs both THUG and NHUG at once, using `iotas` to decide which one.

    Parameters
    ----------
    :param x: Seed positions (N, d)
    :type x: np.ndarray
    :param v: Seed velocities (N, d)
    :type v: np.ndarray
    :param iotas: Array of indicators for THUG (iotas == 1) or SNUG (iotas == 0)
    :type iotas: np.ndarray
    :param B: Number of integration steps
    :type B: int
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
    xnk = np.zeros((N, B+1, d))
    xnk[:, 0] = x
    for b in range(B):
        # Move to midpoint
        xnk[:, b+1] = xnk[:, b] + half_step[:, np.newaxis]*v            # midpoint is (N, 20)
        # Bounce velocities
        v = s[:, np.newaxis]*(v - 2*project(v, gradient(xnk[:, b+1])))  # (N, 20)
        # Move to final point
        xnk[:, b+1] += half_step[:, np.newaxis]*v
    return xnk


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
