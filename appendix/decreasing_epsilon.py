import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma

plt.style.use('seaborn-whitegrid')
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 18
})


d = 2

def lower_bound(n, m, p, d=2, dV=1):
    xi = np.pi**(d/2) / gamma((d/2) + 1)
    return ((dV/xi)*(1 - (p)**(1/m)))**(1/d)

mseq = np.arange(1, 30*d**2, d**2/4)
nseq = np.arange(1, 30*d**2, d**2/4)**3
LVseq = np.arange(1, 30*d**2, d**2/4)**2

p = 0.95

fig, ax = plt.subplots(1, 4, figsize=(12, 3))
for i, LV in enumerate(LVseq[:4]):
    ax[i].plot(mseq, lower_bound(1, mseq.round(0), 1-p, d, dV=LV), linewidth=3, color="blue", label="Lower bound")
    ax[i].set_xlabel("$m$")
    ax[i].set_title("$\mathrm{L}(V)" + f"={int(LV)}$")

ax[0].set_ylabel("Lower bound on $\\varepsilon_m$")
fig.tight_layout()

ax[0].text(
    -0.6, 0.5, f"$d={d}$",
    horizontalalignment='center',
    verticalalignment='center',
    fontsize = "large",
    transform=ax[0].transAxes
)

plt.show()
