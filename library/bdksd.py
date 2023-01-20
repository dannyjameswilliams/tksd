import scipy.optimize as opt
import autograd.numpy as np
from autograd import elementwise_grad as grad

import itertools 

import kgof.util as util
from kgof.kernel import KGauss

# bdKSD where analytic g is on the l1 ball
def bdKSD_l1(X, dlogp, bandwidth=None, theta_init = None, r=1):
    n, d = X.shape

    if bandwidth is None:
        bandwidth = util.meddistance(X, subsample=1000)**2
    K, K_x, K_y, K_xy = kgof_make_Ks_bdksd(X, bandwidth)

    dg, g = polydist_l1ball(X, r)
    
    if theta_init is None:
        theta_init = np.random.randn(d)
    
    obj0 = lambda par: unconstr(par, X, g, dg, dlogp, K, K_x, K_y, K_xy)
    theta_grad = grad(obj0)
    par = opt.minimize(obj0, theta_init, method="BFGS", jac = theta_grad)
    return par.x

# bdKSD where analytic g is on the l2 ball
def bdKSD_l2(X, dlogp, bandwidth=None, theta_init = None, r=1):

    n, d = X.shape

    if bandwidth is None:
        bandwidth = util.meddistance(X, subsample=1000)**2
    K, K_x, K_y, K_xy = kgof_make_Ks_bdksd(X, bandwidth)

    g  = r - np.sqrt(np.sum(X**2, 1))[:, None]
    dg = -X/np.sqrt(np.sum(X**2, 1))[:, None]
    
    if theta_init is None:
        theta_init = np.random.randn(d)
    
    obj0 = lambda par: unconstr(par, X, g, dg, dlogp, K, K_x, K_y, K_xy)
    theta_grad = grad(obj0)
    par = opt.minimize(obj0, theta_init, method="BFGS", jac = theta_grad)
    return par.x

# bdKSD where there is no analytic g - use projections instead 
# (and manually solve for smallest distance)
def bdKSD_proj(X, boundary, dlogp, bandwidth=None, theta_init = None):

    n, d = X.shape

    if bandwidth is None:
        bandwidth = util.meddistance(X, subsample=1000)**2
    K, K_x, K_y, K_xy = kgof_make_Ks_bdksd(X, bandwidth)

    Px = np.empty(X.shape)
    g  = np.empty((n, 1))
    for i in range(n):
        dist = np.sqrt(np.sum((X[i, :] - boundary)**2, 1))
        Px[i, :] = boundary[np.argmin(dist), :]   
        g[i] = np.min(dist)
    
    dg = (X - Px)/g
    
    if theta_init is None:
        theta_init = np.random.randn(d)
    
    obj0 = lambda par: unconstr(par, X, g, dg, dlogp, K, K_x, K_y, K_xy)
    theta_grad = grad(obj0)
    par = opt.minimize(obj0, theta_init, method="BFGS", jac = theta_grad)
    return par.x

# bd-KSD where g is defined in advance (outside of this file)
def bdKSD_fixedg(X, g, dg, dlogp, bandwidth=None, theta_init = None):

    n, d = X.shape

    if bandwidth is None:
        bandwidth = util.meddistance(X, subsample=1000)**2
    K, K_x, K_y, K_xy = kgof_make_Ks_bdksd(X, bandwidth)

    if theta_init is None:
        theta_init = np.random.randn(d)
    
    obj0 = lambda par: unconstr(par, X, g, dg, dlogp, K, K_x, K_y, K_xy)
    theta_grad = grad(obj0)
    par = opt.minimize(obj0, theta_init, method="BFGS", jac = theta_grad)
    return par.x

def kgof_make_Ks_bdksd(X, bandwidth = 1):
    
    n, d = X.shape

    kG = KGauss(bandwidth)
    k1 = kG.eval(X, X)
    k4 = kG.gradXY_sum(X, X)

    k2 = []
    for i in range(d):
        k2.append(kG.gradX_Y(X, X,i))

    k3 = []
    for i in range(d):
        k3.append(kG.gradY_X(X, X,i))

    # organise shapes
    k3 = np.transpose(np.stack(k3,2), (0, 1, 2))

    return k1, np.stack(k2,2), k3, k4

# bdKSD loss function
def unconstr(theta, X, gx, dgx, dlogp, K, K_x, K_y, K_xy):

    p = dlogp(X, theta)
    n, d = X.shape
    gx_row = np.tile(gx.T, (len(gx), 1))

    ggT = gx @ gx.T
    ppT = p @ p.T
    
    term1 = (ggT * ppT * K).sum() 
    term2 = (ggT * (p[None, :, :] * K_x).sum(2)).sum()
    term3 = (gx_row * K * (dgx @ p.T)).sum()
    term6 = (gx_row * (dgx[:, None, :] * K_y).sum(2)).sum() 

    return (term1 + 2*term2 + 2*term3 + 2*term6)


# Get projections from the l1 ball (expensive with high d)    
def polydist_l1ball(Xq, B=1):

    n, d = Xq.shape
    Xq = Xq.T
    t = np.zeros((d, n))
    f = np.zeros(n)

    all1s = np.array(list(itertools.product([-1,1], repeat=d)))/B
    tt = np.zeros((d, len(all1s)))
    ff = np.zeros(len(all1s))

    bj = np.ones(len(all1s)); 
    
    for i in range(n):
        
        for j in range(len(all1s)):

            dist = np.abs((np.dot(all1s[j,:], Xq[:,i]) - bj[j]) / all1s[j,:])
            ff[j] = dist.min()
            idstar = dist.argmin()
              
            tt[:, j] = -all1s[j,:]/np.abs(all1s[j,idstar])

        f[i] = ff.min()
        t[:,i] = tt[:,ff.argmin()]

    return t, f