import autograd.numpy as np

import plotly.graph_objects as go
import matplotlib.pyplot as plt

from library.gram_matrices import kgof_make_Ks_X

def norm_1d(nu, dlogpx, l, K, K_y, Kyt, Kyt_x, Kxtyt):
    
    n = dlogpx.shape[0]
    
    p   = dlogpx[:, l]
    onen = np.ones((n,1))

    ksdpart   = p.T @ K @ p + 2*(p.T @ K_y[l] @ onen)
    tksdpart1 = p.T @ Kyt @ nu + onen.T @ Kyt_x[l] @ nu
    tksdpart2 = nu.T @ Kxtyt @ nu

    return ksdpart[0]/(n**2) + 2*tksdpart1[0]/n + tksdpart2

def analytic_nu(X, dlogpx, l, Kyt, Kyt_x, Kxtyt_inv):
    n = X.shape[0]
    coef = -1/n
    p = dlogpx[:, l]
    onen = np.ones((n, 1))
    return coef * Kxtyt_inv.T @ (p.T @ Kyt + onen.T @ Kyt_x[l]).T
    
def g_(X, Px, dlogp, theta, K, K_y, Kyt, Kyt_x, Kxtyt, Kxtyt_inv, b, xmin=None, xmax=None):

    n, d = X.shape

    if xmin is None:
        xmin = np.vstack((X.min(0), Px.min(0))).min(0)
    if xmax is None:
        xmax = np.vstack((X.max(0), Px.max(0))).max(0)

    if d == 2:
        x_grid = np.linspace(xmin[0], xmax[0], 50)
        y_grid = np.linspace(xmin[1], xmax[1], 50)
        grid = np.array([[x, y] for x in x_grid for y in y_grid])
    elif d == 1:
        grid = np.linspace(xmin[0], xmax[0], 50)[:,None]

    Kxy0, Kxy0_x, Kxty0 = kgof_make_Ks_X(X, Px, grid, b)    
    p = dlogp(X, theta)
    onen = np.ones((n, 1))

    g = lambda l: (p[:, l].T @ Kxy0 + (onen.T @ Kxy0_x[l]).flatten())/n + nu.T @ Kxty0

    gx = np.empty((len(grid), d))
    for l in range(d):
        nu = analytic_nu(X, p, l, Kyt, Kyt_x, Kxtyt_inv)
        norm  = norm_1d(nu, p, l, K, K_y, Kyt, Kyt_x, Kxtyt)
        gx[:, l] = g(l)/norm

        if d==2:
            plot_g_2d(gx[:, l].reshape((len(x_grid), len(y_grid))), X, Px, x_grid, y_grid)
        elif d==1:
            plot_g_1d(gx, X, Px, grid)
    
    return gx

def plot_g_1d(g, X, Px, x0):

    fig, ax = plt.subplots(1, 1, figsize= (8, 6))
    
    ax.scatter(X, np.zeros(len(X)), s=10,c="k", label ="Data")
    ax.scatter(Px, np.zeros(len(Px)), s=100,c="g", label ="Boundary")
    ax.plot(x0, g, label="g(x)",c="r")
    ax.legend()

    ax.set_ylabel("g(x)")
    ax.set_xlabel("x")
    fig.show()

def plot_g_2d(g, X, Px, x0, y0):

    _, d = X.shape

    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=X[:, 0],
                y=X[:, 1],
                z=np.zeros(len(X)),
                mode="markers",
                marker=dict(color="black", size=2),
            ),
            go.Scatter3d(
                x=Px[:, 0],
                y=Px[:, 1],
                z=np.zeros(len(Px)),
                mode="markers",
                marker=dict(color="red", size=4),
            ),
            go.Surface(z=g.T, x=x0, y=y0, opacity=0.5),
        ]
    )
    fig.show(renderer="browser")