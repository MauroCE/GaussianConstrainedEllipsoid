"""
HERE I WILL RUN AN EXPERIMENT FOR SMALL N TO SEE WHAT WE GET.
"""
import pickle
import numpy as np
from ghums_functions import ghums
from smc_functions import smc


if __name__ == "__main__":
    # Settings
    d = 2
    budget = 120_000_000
    eps_final = 1e-5
    eps_init = 1e2
    seed = 1234
    rng = np.random.default_rng(seed=seed)
    step_thug = 0.1
    step_snug = 0.01
    p_thug = 0.8
    verbose = False
    test_functions = [
        lambda x: x[0],
        lambda x: x[1],
    ]
    n_repetitions = 12

    # Grid parameters
    Ns = np.array([1000])
    Ts = np.array([2, 5, 10, 25, 50, 100, 200, 250, 300, 400, 500])
    Ps = {N: (budget/N)/Ts for N in Ns}
    print("-"*28)
    print("NTP budget: ", budget)
    print("\tNs: ", Ns)
    print("\tTs:", Ts)
    print("\tPs: \n", "\n".join([f"\t\t{N}: {Ps[N]}" for N in Ns]))
    print("Budgets: ", np.unique([N*Ps[N]*Ts for N in Ns]))
    print("-" * 28)

    # STORAGE
    OUTS = {
        'ghums': [[] for _ in range(n_repetitions)],
        'smc': [[] for _ in range(n_repetitions)],
        'settings': {
            'Ps': Ps,
            'Ns': Ns,
            'Ts': Ts,
            'd': d,
            'budget': budget,
            'eps_final': eps_final,
            'eps_init': eps_init,
            'seed': seed,
            'p_thug': p_thug,
            'step_thug': step_thug,
            'step_snug': step_snug
        }
    }

    for rep in range(n_repetitions):
        print("Repetition: ", rep)
        for ni, N in enumerate(Ns):
            print("\tN: ", N)
            for ti, T in enumerate(Ts):
                P = int(Ps[N][ti])
                print("\t\tT: ", T, " P: ", P)
                # Generate initial particles
                x0 = rng.normal(size=(N, d))
                v0 = rng.normal(size=(N, d))
                # Generate sequence of epsilons
                epsilons = np.geomspace(start=eps_init, stop=eps_final, num=P+1, endpoint=True, dtype=np.float64)
                # Run GHUMS on this sequence of epsilons
                out_ghums = ghums(x=x0, v=v0, N=N, T=T, funcs=test_functions, epsilons=epsilons, p_thug=p_thug,
                                  step_thug=step_thug, step_snug=step_snug, endpoint=False, adapt_snug=True,
                                  snug_target=0.3, snug_min_step=1e-30, snug_max_step=10.0, min_pm=1e-2, verbose=verbose,
                                  rng=rng)
                OUTS['ghums'][rep].append(out_ghums)
                # Run SMC sampler on this sequence of epsilons
                out_smc = smc(x=x0, epsilons=epsilons, N=N, T=T, step_thug=step_thug, step_snug=step_snug, p_thug=p_thug,
                              snug_target=0.5, min_ap=1e-2, verbose=False, rng=rng, mode='product')
                OUTS['smc'][rep].append(out_smc)
    # Save results
    with open("data/sm_ghums_vs_smc_budget_{}_N1000_rep{}.pkl".format(budget, n_repetitions), "wb") as f:
        pickle.dump(OUTS, f)
