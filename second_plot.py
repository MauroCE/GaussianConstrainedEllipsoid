import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc


# Load data
with open("data/sm_ghums_vs_smc_budget_120000000_N1000.pkl", "rb") as f:
    data = pickle.load(f)

# Cast data into plotting form
Ns, Ts, Ps, d = data['settings']['Ns'], data['settings']['Ts'], data['settings']['Ps'], data['settings']['d']
budget = data['settings']['budget']
# GHUMS
esjd_eff_a_ghums = np.full(shape=(len(Ts), d), fill_value=np.nan)
esjd_eff_b_ghums = np.full(shape=(len(Ts), d), fill_value=np.nan)
# SMC
esjd_eff_smc = np.full(shape=(len(Ts), d), fill_value=np.nan)

for ti in range(len(Ts)):
    # ESJD for GHUMS
    esjd_eff_a_ghums[ti] = data['ghums'][ti]['ESJD-A'] / data['ghums'][ti]['runtime']
    esjd_eff_b_ghums[ti] = data['ghums'][ti]['ESJD-B'] / data['ghums'][ti]['runtime']
    # ESJD for SMC
    esjd_eff_smc[ti] = data['smc'][ti]['esjd'][-1] / data['smc'][ti]['runtime']

# ESJD-A
P_axis = False
esjd_type = 'B'
esjd_array = esjd_eff_a_ghums if esjd_type == 'A' else esjd_eff_b_ghums
esjd_label = r"$\mathregular{ESJD_A/s}$" if esjd_type == 'A' else r"$\mathregular{ESJD_B/s}$"

rc('font', **{'family': 'STIXGeneral'})
fig, ax = plt.subplots(ncols=d, figsize=(8, 4), sharex='col', sharey=True)
for cix in range(d):
    ax[cix].plot(Ts, esjd_array[:, cix], marker='o', label='GHUMS', c='lightcoral', mec='maroon')
    ax[cix].plot(Ts, esjd_eff_smc[:, cix], marker='o', label='SMC', c='#C9E23C', mec='#5D6A10')
    if P_axis:
        ax2 = ax[cix].secondary_xaxis('top', functions=(lambda ts: (budget/Ns[0]) / ts,
                                                             lambda ps: (budget/Ns[0]) / ps))
        ax2.set_xlabel(r"$\mathregular{P}$")
    ax[cix].set_xscale('log')
    ax[cix].set_yscale('log')
    ax[cix].set_xticks(Ts)
    ax[cix].set_yticks(np.geomspace(start=1e-11, stop=1e-1, endpoint=True, num=11))
    ax[cix].grid(True, color='gainsboro')
    if cix == 0:
        ax[cix].set_ylabel(esjd_label)
for cix in range(d):
    ax[cix].set_xlabel(r"$\mathregular{T}$")
    ax[cix].set_title(f"Coordinate {cix+1}")
plt.legend()
plt.tight_layout()
plt.show()
