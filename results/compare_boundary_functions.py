import autograd.numpy as np
import matplotlib.pyplot as plt
from autograd import elementwise_grad as grad

import sys
import os
sys.path.append(os.getcwd())

from kgof import util

from library.gram_matrices import kgof_make_Ks_X, kgof_make_Ks
from library.bdksd import bdKSD_l2
from library.tksd_v_statistic import tksd, cholesky_inv
from library.tksd_g import norm_1d, analytic_nu
from library.truncsm import truncsm_l2


plt.style.use('seaborn-whitegrid')
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Helvetica",
    "font.size": 18
})

# Functions to calculate discriminatory  function for bd-KSD and TKSD
def bdKSD_h(X, x, theta, dlogp, b):
    KX, dK_xX, _ = kgof_make_Ks_X(X, np.empty((0, 1)), x, b)
    return (dlogp(X, theta).T @ KX).sum(0) + dK_xX[0].sum(0)

def TKSD_g(X, x, Px, theta, dlogp, b):
    K, K_y, Kyt, Kyt_x, Kxtyt = kgof_make_Ks(Xt, Px, b)
    Kxy0, Kxy0_x, Kxty0 = kgof_make_Ks_X(X, Px, x, b)   
    Kxtyt_inv = cholesky_inv(Kxtyt, 1e-3)

    onen = np.ones((n, 1))
    p = dlogp(X, theta)
    
    nu = analytic_nu(X, p, 0, Kyt, Kyt_x, Kxtyt_inv)
    norm  = norm_1d(nu, p, 0, K, K_y, Kyt, Kyt_x, Kxtyt)
    g = -((p.T @ Kxy0 + (onen.T @ Kxy0_x).flatten())/n + nu.T @ Kxty0)/norm

    return g[0]

if __name__ == "__main__":

    # Fix random seed
    np.random.seed(1)

    # Simulate truncated data from 1D Gaussian
    Xall = np.random.normal(0, 1, 50)[:, None]
    Xt   = Xall[np.sqrt(np.sum(Xall**2, 1)) < 1]
    Px   = np.array([[-1], [1]])
    n    = len(Xt)

    # Set bandwidth as default
    bandwidth = util.meddistance(Xt, subsample=1000)**2

    # Set up parametric score functions
    logp  = lambda x, theta: -((x - theta)**2).sum(1)/2
    dlogp = grad(logp)

    # bd-KSD and TKSD need estimates of theta for their discriminatory functions
    tksd_theta = tksd(Xt, Px, dlogp, 1, plot_g=False)
    bdksd_theta = bdKSD_l2(Xt, dlogp, bandwidth)
    truncsm_theta = truncsm_l2(Xt, dlogp)

    # Set up grid for plotting
    grid = np.linspace(Px[0][0], Px[1][0], 150)[:, None]

    tksd_theta = -0.75
    bdksd_theta = -0.75
    truncsm_theta = -0.75

    # tksd_theta = 0
    # bdksd_theta = 0
    # truncsm_theta = 0

    # g (TKSD)
    g_tksd = TKSD_g(Xt, grid, Px, tksd_theta, dlogp, bandwidth)
    
    # g and h which multiply (bdKSD)
    g_dist = 1 - np.linalg.norm(grid, 1, 1)[:, None]
    hbd = bdKSD_h(Xt, grid, bdksd_theta, dlogp, bandwidth)[:, None]
    bd2 = g_dist * hbd


    # Plot output
    fig, axes = plt.subplots(1, 3, figsize= (10, 2.3))

    for ax in axes:
        ax.set_xlim(-2.2, 2.2)
        ax.set_ylim(-1.05, 1.2)
        ax.scatter(Xall, np.zeros(len(Xall)), s=8, label = "Unobserved", c="#a1a1a1")
        ax.scatter(Xt, np.zeros(len(Xt)), s=8, label = "Observed", c="k")
        ax.vlines(1, -3, 3, "k", linestyles="--", alpha=0.5, label = "Truncation")
        ax.vlines(-1, -3, 3, "k", linestyles="--", alpha=0.5)
        ax.set_xlabel("$x$")
        # ax.set_xticks([])


    axes[1].set_yticklabels([])
    axes[2].set_yticklabels([])

    # Scale g functions so that they all fit in the same plot
    axes[0].set_ylabel("$f(x)$")
    axes[2].plot(grid, g_tksd/(3*g_tksd.std()), label = "TKSD", linewidth=3, c="b")
    axes[2].set_title("TKSD")
    axes[2].scatter(tksd_theta, 0, c="m",s=120, marker="*")
    axes[2].text(tksd_theta, 0.22, r"$\hat{\mu}$", c="m", fontdict={"size":26})

    axes[0].plot(grid, g_dist/(3.1*g_dist.std()), label = "truncSM", linewidth=3, c="r")
    axes[0].set_title("truncSM")
    axes[0].scatter(truncsm_theta, 0, c="m",s=120, marker="*")
    axes[0].text(truncsm_theta+0.08, 0.22, r"$\hat{\mu}$", c="m", fontdict={"size":26})

    axes[1].plot(grid, bd2/(3*bd2.std()), label = "bd-KSD", linewidth=3, c="g")
    axes[1].set_title("bd-KSD")
    axes[1].scatter(bdksd_theta, 0, c="m",s=120, marker="*")
    axes[1].text(bdksd_theta, 0.22, r"$\hat{\mu}$", c="m", fontdict={"size":26})

    plt.show()
