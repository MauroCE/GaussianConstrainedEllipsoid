"""
User provides a fixed final tolerance `eps_final` and we use the budget in different ways to get there.
"""
import numpy as np
from ellipsoid_functions import constraint


if __name__ == "__main__":
    # Settings
    d = 2
    budget = 1_500_000
    eps_final = 1e-5
    eps_init = 1e2
    seed = 1234
    rng = np.random.default_rng(seed=seed)

    # Grid parameters
    Ps = [25, 50, 100]
    Ns = [100, 500, 1000]
    Ts = (budget / np.outer(Ps, Ns)).astype(np.int32)
    print("-"*28)
    print("NTP budget: ", budget)
    print("\tPs: ", Ps)
    print("\tNs: ", Ns)
    print("\tTs: \n", Ts)
    print("-" * 28)

    for pi, P in enumerate(Ps):
        print("P: ", P)
        for ni, N in enumerate(Ns):
            print("\tN: ", N, " T: ", Ts[pi, ni])
            T = Ts[pi, ni]  # Grab number of integration steps
            # Generate initial particles
            x0 = rng.normal(size=(N, d))
            v0 = rng.normal(size=(N, d))
            # Generate sequence of epsilons
            epsilons = np.geomspace(start=eps_init, stop=eps_final, num=P+1, endpoint=True, dtype=np.float64)
            # Run GHUMS on this sequence of epsilons
            # Run SMC sampler on this sequence of epsilons



