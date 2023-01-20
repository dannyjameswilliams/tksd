import autograd.numpy as np
from autograd import elementwise_grad as grad

import scipy.optimize as opt
import itertools 
import autograd.numpy as np

# TruncSM where analytic g is on the l1 ball
def truncsm_l1(X, dlogp, r=1, theta_init=None):
    n, d = X.shape

    dg, g = polydist_l1ball(X, r)

    # np.random.seed(1)
    if theta_init is None:
        theta_init = np.random.randn(d)
    
    ddlogp = grad(dlogp)

    fun = lambda theta: obj(X, theta, g[:, None], dg.T, dlogp, ddlogp)
    theta_grad = grad(fun)
    par = opt.minimize(fun, theta_init, method="BFGS", jac = theta_grad)
    return par.x

# TruncSM where analytic g is on the l2 ball
def truncsm_l2(X, dlogp, theta_init=None, r=1):
    
    n, d = X.shape

    if theta_init is None:
        theta_init = np.random.randn(d)

    g  = r - np.sqrt(np.sum(X**2, 1))[:, None]
    dg = -X/np.sqrt(np.sum(X**2, 1))[:, None]
    
    ddlogp = grad(dlogp)
    
    fun = lambda theta: obj(X, theta, g, dg, dlogp, ddlogp)
    theta_grad = grad(fun)
    par = opt.minimize(fun, theta_init, method="BFGS", jac = theta_grad)
    return par.x


# TruncSM where there is no analytic g - use projections instead 
# (and manually solve for smallest distance)
def truncsm_proj(X, boundary, dlogp, theta_init=None):
    
    n, d = X.shape

    if theta_init is None:
        theta_init = np.random.randn(d)

    Px = np.empty(X.shape)
    g  = np.empty((n, 1))
    for i in range(n):
        dist = np.sqrt(np.sum((X[i, :] - boundary)**2, 1))
        Px[i, :] = boundary[np.argmin(dist), :]   
        g[i] = np.min(dist)
    
    dg = (X - Px)/g

    ddlogp = grad(dlogp)

    fun = lambda theta: obj(X, theta, g, dg, dlogp, ddlogp)
    theta_grad = grad(fun)
    par = opt.minimize(fun, theta_init, method="BFGS", jac = theta_grad)
    return par.x

# TruncSM where g is defined in advance (outside of this file)
def truncsm_fixedg(X, g, dg, dlogp, theta_init=None):
    
    n, d = X.shape

    # np.random.seed(1)
    if theta_init is None:
        theta_init = np.random.randn(d)
    
    ddlogp = grad(dlogp)

    fun = lambda theta: obj(X, theta, g, dg, dlogp, ddlogp)
    theta_grad = grad(fun)
    par = opt.minimize(fun, theta_init, method="BFGS", jac = theta_grad)
    return par.x

# Objective function for all variations
def obj(X, theta, gx, dgx, dlogp, ddlogp):
    
    p  = dlogp(X, theta)
    dp = ddlogp(X, theta)

    f = (np.mean(np.sum(p**2*gx, 1)) + 
        2*np.mean(np.sum(dp*gx,  1)) +  
        2*np.mean(np.sum(p*dgx,  1)))
    
    return f

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