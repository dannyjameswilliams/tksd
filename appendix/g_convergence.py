import autograd.numpy as np
import matplotlib.pyplot as plt
import matplotlib

from autograd import elementwise_grad as grad
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from kgof import util

import sys
import os
sys.path.append(os.getcwd())

from library.tksd_v_statistic import tksd, cholesky_inv
from library.gram_matrices import kgof_make_Ks_all, kgof_make_Ks_X

plt.style.use('seaborn-whitegrid')
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Helvetica",
    "font.size": 18
})

# Organise polygon (clockwise)
def organise_poly(corners, centroid):
    
    ang = np.zeros(corners.shape[0])
    for Pi in range(corners.shape[0]):
        Pxi = corners[Pi,:]

        adj = np.abs(Pxi[0] - centroid[0])
        hyp = np.linalg.norm(Pxi - centroid, 2)
        ang[Pi] = np.arccos(adj/hyp)
        
        pos = Pxi - centroid
        if pos[0] > 0 and pos[1] > 0:
            ang[Pi] = np.pi - ang[Pi]
        elif pos[0] > 0 and pos[1] < 0:
            ang[Pi] = np.pi + ang[Pi]
        elif pos[0] < 0 and pos[1] < 0:
            ang[Pi] = 2*np.pi - ang[Pi]
        
    ind = np.argsort(ang)
    return corners[ind, :]


def simulate(f, n=30, d=2, m=10, seed=2, q=2, mu = None, sigma=1):


    # Get mean and variance for simulation
    np.random.seed(seed)
    if mu is None:
        mu = np.ones(d)/d
    Sigma = np.identity(d)*sigma

    # Use a grid, with matplotlib contour function to get boundary points for a specific shape
    x_ = np.linspace(-2, 2, 200)
    y_ = np.linspace(-2, 2, 200)
    xx, yy = np.meshgrid(x_, y_)
    grid = np.c_[xx.ravel(), yy.ravel()]
    z = f(grid).reshape(xx.shape)

    plt_backend = matplotlib.get_backend()
    c = plt.contour(xx, yy, z, [0, 9999], alpha=0);
    matplotlib.use(plt_backend)
    c0 = c.allsegs[0][0]

    # Use polygon library and organise_poly function above to use for truncation
    poly = Polygon(organise_poly(c0, np.zeros(d)))

    # Simulate until we have n truncated samples
    Xt = np.empty((0, d))
    while np.shape(Xt)[0] < n:
        X  = np.random.multivariate_normal(mu, Sigma, 10000)
        Xin = np.zeros((0, 2))
        for i in range(len(X)):
            if poly.contains(Point(X[i, :])):
                Xin = np.vstack((Xin, X[i, :]))
        Xt = np.vstack((Xt, Xin))
    Xt = Xt[:n, :]
    
    # Subset boundary to desired number of points
    ind = np.random.permutation(len(c0))[:m]

    return Xt, c0[ind, :]

# Functions used to calculate g(x)
def norm_1d(nu, dlogpx, l, K, K_y, K_xy, Kyt, Kyt_x, Kxtyt):
    n = dlogpx.shape[0]
    p   = dlogpx[:, l]
    onen = np.ones((n,1))
    ksdpart   = p.T @ K @ p + 2*(p.T @ K_y[l] @ onen) + onen.T @ K_xy @ onen
    tksdpart1 = p.T @ Kyt @ nu + onen.T @ Kyt_x[l] @ nu
    tksdpart2 = nu.T @ Kxtyt @ nu
    return (ksdpart[0][0]/(n**2) + 2*tksdpart1[0]/n + tksdpart2)**0.5

def analytic_nu(X, dlogpx, l, Kyt, Kyt_x, Kxtyt_inv):
    n = X.shape[0]
    coef = -1/n
    p = dlogpx[:, l]
    onen = np.ones((n, 1))
    return coef * Kxtyt_inv.T @ (p.T @ Kyt + onen.T @ Kyt_x[l]).T

def TKSD_g(X, x, Px, theta, dlogp, b, l=0):
    n = len(X)
    K, K_y, K_xy, Kyt, Kyt_x, Kxtyt = kgof_make_Ks_all(X, Px, b)
    Kxy0, Kxy0_x, Kxty0 = kgof_make_Ks_X(X, Px, x, b)   
    Kxtyt_inv = cholesky_inv(Kxtyt, 1e-3)
    onen = np.ones((n, 1))
    p = dlogp(X, theta)
    nu = analytic_nu(X, p, 0, Kyt, Kyt_x, Kxtyt_inv)
    norm  = norm_1d(nu, p, 0, K, K_y, K_xy, Kyt, Kyt_x, Kxtyt)
    g = -((p[:, l].T @ Kxy0 + (onen.T @ Kxy0_x[l]).flatten())/n + nu.T @ Kxty0)/norm
    return g[0]

# Function that does most of the work - given a boundary shape, fit TKSD, get g(x) and plot it
def plot_increasing_m(f, f2=None):

    if f2 is None:
        f2 = f

    np.random.seed(1)

    mseq = [10, 50, 150, 300]
    fig, ax = plt.subplots(1, len(mseq), figsize=(11, 2.5))

    for i, m in enumerate(mseq):

        X, Px = simulate(f, n=150, d=2, m = m, seed=1)

        logp  = lambda x, theta: -((x - theta.flatten())**2).sum(1)/(2)
        dlogp = grad(logp, 0)
        theta = tksd(X, Px, dlogp)

        x_ = np.linspace(-2, 2, 200)
        y_ = np.linspace(-2, 2, 200)
        xx, yy = np.meshgrid(x_, y_)
        grid = np.c_[xx.ravel(), yy.ravel()]

        g = TKSD_g(X, grid, Px, theta, dlogp, util.meddistance(X)**2)

        # slightly bigger f (f2) just for limiting plot of g
        _, Px2 = simulate(f2, n=100, d=2, m = m, seed=1)

        poly = Polygon(organise_poly(Px2, np.zeros(2)))
        gin = np.empty(len(grid), dtype=bool)
        for j in range(len(grid)):
            gin[j] = poly.contains(Point(grid[j, :]))
        g[~gin] = np.nan
        g = g.reshape(xx.shape)

        # plt.imshow(g, extent = (-2, 2, -2, 2))
        ax[i].scatter(X[:, 0], X[:, 1], c="k", s=3, alpha=0.25)
        ax[i].scatter(Px[:, 0], Px[:, 1], c="r", s=15)

        levels = np.quantile(g[~np.isnan(g)], np.arange(0.15, 1, 0.15))
        levels = np.append(levels, 0).round(5)
        levels.sort()

        c = ax[i].contour(xx, yy, g, cmap="viridis", levels=levels)

        ax[i].set_xlim(-2, 2)
        ax[i].set_ylim(-2, 2)
        ax[i].set_title("$m = " + str(m) + "$")
        ax[i].set_xticklabels([])
        ax[i].set_yticklabels([])

        norm= matplotlib.colors.Normalize(vmin=c.cvalues.min(), vmax=c.cvalues.max())
        sm = plt.cm.ScalarMappable(norm=norm, cmap = c.cmap)
        sm.set_array([])
        fmt = matplotlib.ticker.ScalarFormatter(useMathText=True)
        fmt.set_powerlimits((0, 0))
        
        ticks = c.levels[[0, -1]]
        ticks = np.append(ticks, 0)
        ticks.sort()
        if abs(ticks[0]) < 5e-3:
            ticks = ticks[1:]

        cb = fig.colorbar(sm, orientation="horizontal", ax=ax[i], ticks=ticks, format=fmt, pad=0.05)
        cb.ax.tick_params(labelsize=11)
        cb.ax.xaxis.get_offset_text().set(size=11) 

    plt.show()

if __name__ == "__main__":
    
    # Circle
    def circle(x, r):
        return x[:, 0]**2 + x[:, 1]**2 - r

    plot_increasing_m(lambda x: circle(x, 1), lambda x: circle(x, 1.1))

    # l1 ball
    def l1(x, r):
        return np.linalg.norm(x, 1, 1) - r

    plot_increasing_m(lambda x: l1(x, 1), lambda x: l1(x, 1.1))

    # Heart
    def heart(x, r):
        return x[:, 0]**2 + (x[:, 1] - (x[:, 0]**2)**(1/3))**2 - r

    plot_increasing_m(lambda x: heart(x, 1), lambda x: heart(x, 1.1))