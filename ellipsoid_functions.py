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
    :rtype: np.ndarray
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


# ---------- FUNCTIONS ON 3D TENSORS -----------
def constraint3d(x):
    """Expects x to have shape (n_particles, n_int_steps + 1, d). Output has shape (n_particles, n_int_steps+1)."""
    d = x.shape[2]
    return (d/4)*np.log(10) - (d//2)*np.log(2*np.pi) - 0.5*np.multiply(
        np.tile(np.array([1., 10.]), (len(x), x.shape[1], d//2)), x**2).sum(axis=2) - level_set(d)


def log_post3d(x, epsilon):
    """Computes log posterior density on array of shape (n_particles, n_int_steps+1, d)."""
    log_density_values = np.full(x.shape[:2], -np.inf)
    flag = np.abs(constraint3d(x)) <= epsilon   # (N, B+1)
    log_density_values[flag] = -0.5*np.sum(x[flag]**2, axis=1) - np.log(2*np.pi)
    return log_density_values
