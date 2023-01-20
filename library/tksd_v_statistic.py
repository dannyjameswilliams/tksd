import autograd.numpy as np
from autograd import elementwise_grad as grad

from scipy.optimize import minimize
from scipy.linalg import solve_triangular

import kgof.util as util

from library.gram_matrices import kgof_make_Ks

def obj_vec(X, theta, dlogp, K, K_y, Kyt, Kyt_x, Kxtyt_inv):
    n, d = X.shape
    dlogpx = dlogp(X, theta)
    onen = np.ones(n)

    # ksd part
    ksd1 = ((dlogpx @ dlogpx.T) * K).sum()
    ksd2 = 2*np.array([dlogpx[:, l].T @ K_y[l] @ onen for l in range(d)]).sum()

    # tksd part
    tksd1 = (dlogpx.T @ Kyt)
    tksd2 = Kyt_x

    tksd3_comb = ((tksd1.T @ tksd1 + 2*(tksd2 @ tksd1)) * Kxtyt_inv).sum()
    return ((ksd1+ksd2)/(n**2) - tksd3_comb/(n**2))


def analytic_nu(X, dlogpx, l, Kyt, Kyt_x, Kxtyt_inv):
    n = X.shape[0]
    coef = -1/n
    p = dlogpx[:, l]
    onen = np.ones((n, 1))
    return coef * Kxtyt_inv.T @ (p.T @ Kyt + onen.T @ Kyt_x[l]).T
    
def cholesky_inv(A, reg):
    L  = np.linalg.cholesky(A + reg*np.eye(len(A)))
    Linv = solve_triangular(L.T, np.eye(len(L)))
    return Linv @ Linv.T
    
def tksd(X, Px, dlogp, bandwidth=None, theta_init=None, plot_g = False):

    # Get shapes
    n, d   = X.shape

    # Kernel bandwidth is default median distance
    if bandwidth is None:
        bandwidth = util.meddistance(X, subsample=1000)**2

    # Kernel matrices
    K, K_y, Kyt, Kyt_x, Kxtyt = kgof_make_Ks(X, Px, bandwidth)
    Kxtyt_inv = cholesky_inv(Kxtyt, 1e-3)

    # precompute some stuff
    Kyt_x0 = np.stack(Kyt_x, 2).sum(0)

    # Initial conditions are default random
    if theta_init is None:
        theta_init = np.random.randn(d)
    n_p = len(theta_init)

    # Objective as a function of theta only
    obj0 = lambda par: obj_vec(X, par, dlogp, K, K_y, Kyt, Kyt_x0, Kxtyt_inv)
    theta_grad = grad(obj0)

    # scipy minimize
    opt = dict(maxiter=1e5, disp=False)
    par = minimize(obj0, theta_init, options=opt, method="BFGS", jac = theta_grad)
    theta = par.x

    if plot_g:
        from library.tksd_g import g_
        g = "must be d<=2 for plot_g=True"
        if d == 1:
            g = g_(X, Px, dlogp, theta, K, K_y, Kyt, Kyt_x, Kxtyt, Kxtyt_inv, bandwidth);
        if d == 2:
            g = g_(X, Px, dlogp, theta, K, K_y, Kyt, Kyt_x, Kxtyt, Kxtyt_inv, bandwidth);
        return theta, g
    else:
        return theta

