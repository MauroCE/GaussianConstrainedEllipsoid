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
esjd_eff_a_ghums = np.full(shape=(len(Ns), len(Ts), d), fill_value=np.nan)
esjd_eff_b_ghums = np.full(shape=(len(Ns), len(Ts), d), fill_value=np.nan)
esjd_eff_smc = np.full(shape=(len(Ns), len(Ts), d), fill_value=np.nan)
print("Ns: ", Ns)
print("Ps: ", Ps)
print("Ts: ", Ts)

for ni in range(len(Ns)):
    for ti in range(len(Ts)):
        # ESJD for GHUMS
        esjd_eff_a_ghums[ni, ti] = data['ghums'][ni*len(Ts) + ti]['ESJD-A'] / data['ghums'][ni*len(Ts) + ti]['runtime']
        esjd_eff_b_ghums[ni, ti] = data['ghums'][ni*len(Ts) + ti]['ESJD-B'] / data['ghums'][ni*len(Ts) + ti]['runtime']
        # ESJD for SMC
        esjd_eff_smc[ni, ti] = data['smc'][ni*len(Ts) + ti]['esjd'][-1] / data['smc'][ni*len(Ts) + ti]['runtime']

# ESJD-A
P_axis = False
esjd_type = 'B'
esjd_array = esjd_eff_a_ghums if esjd_type == 'A' else esjd_eff_b_ghums
esjd_label = f"ESJD-A/s" if esjd_type == 'A' else f"ESJD-B/s"

rc('font', **{'family': 'STIXGeneral'})
fig, ax = plt.subplots(nrows=len(Ns), ncols=d, figsize=(8, 8), sharex='col', sharey=True)
for rix in range(len(Ns)):
    for cix in range(d):
        ax[rix, cix].plot(Ts, esjd_array[rix, :, cix], marker='o', label='ghums')
        ax[rix, cix].plot(Ts, esjd_eff_smc[rix, :, cix], marker='o', label='smc')
        if P_axis:
            ax2 = ax[rix, cix].secondary_xaxis('top', functions=(lambda ts: (budget/Ns[rix]) / ts,
                                                                 lambda ps: (budget/Ns[rix]) / ps))
            ax2.set_xlabel(r"$\mathregular{P}$")
        ax[rix, cix].set_xscale('log')
        ax[rix, cix].set_yscale('log')
        ax[rix, cix].grid(True, color='gainsboro')
        if cix == 0:
            ax[rix, cix].set_ylabel(esjd_label)
for cix in range(d):
    ax[-1, cix].set_xlabel(r"$\mathregular{T}$")
plt.legend()
plt.tight_layout()
plt.savefig("images/ghums_vs_smc_efficiency_type{}.png".format(esjd_type.lower()), dpi=300)
plt.show()
