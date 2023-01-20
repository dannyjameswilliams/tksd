import numpy as np
import matplotlib.pyplot as plt

plt.style.use('seaborn-whitegrid')
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Helvetica",
    "font.size": 18
})


def eps_m(n, m, p, dV=1, xi=1):
    return (dV/xi)*(1 - (p/n)**(1/m))


mseq = np.arange(1, 200, 5)
nseq = np.arange(1, 200, 5)**4

p = 0.95

fig, ax = plt.subplots(1, 4, figsize=(12, 3))
for i, n in enumerate(nseq[:4]):
    ax[i].plot(mseq, eps_m(n, mseq, 1-p), linewidth=3)
    ax[i].set_xlabel("$m$")
    ax[i].set_ylabel("$\\varepsilon_m$")
    ax[i].set_title("$n_{\\varepsilon_m}" + f"={n}$")
    ax[i].set_ylim(0, 1)
fig.tight_layout()

plt.savefig("decreasing_epsilon.pdf")