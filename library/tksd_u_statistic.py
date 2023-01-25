import autograd.numpy as np
from autograd import elementwise_grad as grad

from scipy.optimize import minimize
from scipy.linalg import solve_triangular

import kgof.util as util

from library.gram_matrices import kgof_make_Ks_all

def obj_vec(X, theta, dlogp, K, K_y, K_xy, Kyt, Kyt_x, Kxtyt_inv):
    n, d = X.shape
    dlogpx = dlogp(X, theta)
    onen = np.ones(n)

    # ksd part
    ksd1 = ((dlogpx @ dlogpx.T) * K).sum()
    ksd2 = 2*np.array([dlogpx[:, l].T @ K_y[l] @ onen for l in range(d)]).sum()
    ksd3 = K_xy

    # tksd part
    tksd1 = (dlogpx[:, None, :] * Kyt[:, :, None])
    u = (tksd1 + np.stack(Kyt_x, 2))
    u3 = np.zeros((n, n))
    for l in range(d):
        u3 += u[:,:,l] @ Kxtyt_inv @ u[:,:,l].T

    u3 -= np.diag(u3)*np.eye(n)

    return ksd1 + ksd2 + ksd3 - u3.sum()

def cholesky_inv(A, reg):
    L  = np.linalg.cholesky(A + reg*np.eye(len(A)))
    Linv = solve_triangular(L.T, np.eye(len(L)))
    return Linv @ Linv.T
    
def tksd(X, Px, dlogp, bandwidth=None, theta_init=None, plot_g=False):

    # Get shapes
    n, d   = X.shape

    # Kernel bandwidth is default median distance
    if bandwidth is None:
        bandwidth = util.meddistance(X, subsample=1000)**2
        # print(f"bandwidth={bandwidth.round(3)}")

    # Kernel matrices
    K, K_y, K_xy, Kyt, Kyt_x, Kxtyt = kgof_make_Ks_all(X, Px, bandwidth)
    Kxtyt_inv = cholesky_inv(Kxtyt, 1e-3)

    # for U-statistic, i \neq j, so make diagonal entries zero for samples n
    np.fill_diagonal(K, 0)
    np.fill_diagonal(K_xy, 0)
    for l in range(d):
        np.fill_diagonal(K_y[l], 0)

    # precompute some stuff
    K_xy0  = K_xy.sum()*d

    # Initial conditions are default random
    if theta_init is None:
        theta_init = np.random.randn(d)

    # Objective as a function of theta only
    obj0 = lambda par: obj_vec(X, par, dlogp, K, K_y, K_xy0, Kyt, Kyt_x, Kxtyt_inv)
    theta_grad = grad(obj0)

    # scipy minimize
    opt = dict(maxiter=1e10, disp=False)
    par = minimize(obj0, theta_init, options=opt, method="BFGS", jac = theta_grad)
    theta = par.x

    if plot_g:
        from library.tksd_g import g_
        g = "must be d<=2 for plot_g=True"
        if d <= 2:
            g = g_(X, Px, dlogp, theta, K, K_y, Kyt, Kyt_x, Kxtyt, Kxtyt_inv, bandwidth);
        return theta, g
    else:
        return theta
