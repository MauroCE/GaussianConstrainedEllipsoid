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
esjd_eff_a = np.full(shape=(len(Ns), len(Ps), d), fill_value=np.nan)
esjd_eff_b = np.full(shape=(len(Ns), len(Ps), d), fill_value=np.nan)
print("Ns: ", Ns)
print("Ps: ", Ps)
print("Ts: ", Ts)

for pi in range(len(Ps)):
    for ni in range(len(Ns)):
        esjd_eff_a[ni, pi] = data['ghums'][pi*len(Ns) + ni]['ESJD-A'] / data['ghums'][pi*len(Ns) + ni]['runtime']
        esjd_eff_b[ni, pi] = data['ghums'][pi*len(Ns) + ni]['ESJD-B'] / data['ghums'][pi*len(Ns) + ni]['runtime']


# secondary_x_axes = np.full(shape=(len(Ns), d), fill_value=np.nan, dtype=object)
rc('font', **{'family': 'STIXGeneral'})
fig, ax = plt.subplots(nrows=len(Ns), ncols=d, figsize=(8, 8), sharex='col', sharey=True)
for rix in range(len(Ns)):
    for cix in range(d):
        ax[rix, cix].plot(Ts[:, rix], esjd_eff_a[rix, :, cix], marker='o')
        # T_to_P = lambda T_values: (budget // Ns[rix]) / T_values
        # P_to_T = lambda P_values: (budget // Ns[rix]) / P_values
        # secondary_x_axes[rix, cix] = ax[rix, cix].secondary_xaxis('top', functions=(T_to_P, P_to_T))
        # if rix == 0:
        #     secondary_x_axes[rix, cix].set_xlabel("P")
        # secondary_x_axes[rix, cix].set_xscale('log')
        ax[rix, cix].set_xscale('log')
        ax[rix, cix].set_yscale('log')
        ax[rix, cix].grid(True, color='gainsboro')
        if cix == 0:
            ax[rix, cix].set_ylabel(f"ESJD-A/s")
for cix in range(d):
    ax[-1, cix].set_xlabel(r"$\mathregular{T}$")
plt.tight_layout()
plt.show()
