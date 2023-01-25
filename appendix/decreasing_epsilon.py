import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma

plt.style.use('seaborn-whitegrid')
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Helvetica",
    "font.size": 18
})


d = 2

def eps_m(n, m, p, d=2, dV=1):
    xi = np.pi**(d/2) / gamma((d/2) + 1)
    return ((dV/xi)*(1 - (p/n)**(1/m)))**(1/d)


mseq = np.arange(1, 30*d**2, d**2/4)
nseq = np.arange(1, 30*d**2, d**2/4)**3

p = 0.95

fig, ax = plt.subplots(1, 4, figsize=(12, 3))
for i, n in enumerate(nseq[:4]):
    ax[i].plot(mseq, eps_m(int(n), mseq.round(0), 1-p, d), linewidth=3)
    ax[i].set_xlabel("$m$")
    ax[i].set_ylabel("$\\varepsilon_m$")
    ax[i].set_title("$n_{\\varepsilon_m}" + f"={int(n)}$")
fig.tight_layout()

ax[0].text(
    -0.6, 0.5, f"$d={d}$",
    horizontalalignment='center',
    verticalalignment='center',
    fontsize = "large",
    transform=ax[0].transAxes
)

plt.savefig(f"epsilon_d{d}.pdf", bbox_inches="tight")
plt.show()