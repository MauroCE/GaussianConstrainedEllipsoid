import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc


# Load data
with open("data/ghums_vs_smc.pkl", "rb") as f:
    data = pickle.load(f)

# Cast data into plotting form
Ns, Ts, Ps, d = data['settings']['Ns'], data['settings']['Ts'], data['settings']['Ps'], data['settings']['d']
budget = data['settings']['budget']
esjd_eff_a_ghums = np.full(shape=(len(Ns), len(Ps), d), fill_value=np.nan)
esjd_eff_b_ghums = np.full(shape=(len(Ns), len(Ps), d), fill_value=np.nan)
esjd_eff_a_smc = np.full(shape=(len(Ns), len(Ps), d), fill_value=np.nan)
esjd_eff_b_smc = np.full(shape=(len(Ns), len(Ps), d), fill_value=np.nan)
print("Ns: ", Ns)
print("Ps: ", Ps)
print("Ts: ", Ts)

for pi in range(len(Ps)):
    for ni in range(len(Ns)):
        # ESJD for GHUMS
        esjd_eff_a_ghums[ni, pi] = data['ghums'][pi*len(Ns) + ni]['ESJD-A'] / data['ghums'][pi*len(Ns) + ni]['runtime']
        esjd_eff_b_ghums[ni, pi] = data['ghums'][pi*len(Ns) + ni]['ESJD-B'] / data['ghums'][pi*len(Ns) + ni]['runtime']
        # ESJD for SMC
        esjd_eff_a_smc[ni, pi] = data['smc'][pi*len(Ns) + ni]['esjd'][-1] / data['smc'][pi*len(Ns) + ni]['runtime']
        esjd_eff_b_smc[ni, pi] = data['smc'][pi*len(Ns) + ni]['esjd'][-1] / data['smc'][pi*len(Ns) + ni]['runtime']


# secondary_x_axes = np.full(shape=(len(Ns), d), fill_value=np.nan, dtype=object)
rc('font', **{'family': 'STIXGeneral'})
fig, ax = plt.subplots(nrows=len(Ns), ncols=d, figsize=(8, 8), sharex='col', sharey=True)
for rix in range(len(Ns)):
    for cix in range(d):
        ax[rix, cix].plot(Ts[:, rix], esjd_eff_a_ghums[rix, :, cix], marker='o', label='ghums')
        ax[rix, cix].plot(Ts[:, rix], esjd_eff_a_smc[rix, :, cix], marker='o', label='smc')
        ax[rix, cix].set_xscale('log')
        ax[rix, cix].set_yscale('log')
        ax[rix, cix].grid(True, color='gainsboro')
        if cix == 0:
            ax[rix, cix].set_ylabel(f"ESJD-A/s")
for cix in range(d):
    ax[-1, cix].set_xlabel(r"$\mathregular{T}$")
plt.legend()
plt.tight_layout()
plt.show()
