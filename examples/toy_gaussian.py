import autograd.numpy as np
from autograd import elementwise_grad as grad

import time

import sys
import os
sys.path.append(os.getcwd())

from library.truncsm import truncsm_l1, truncsm_l2
from library.bdksd import bdKSD_l1, bdKSD_l2

# Either the U-statistic or V-statistic form of TKSD (comment out)
from library.tksd_v_statistic import tksd
# from library.tksd_u_statistic import tksd

def simulate(n=30, d=2, m=10, r=1, seed=2, q=2, mu = None, sigma=1):

    np.random.seed(seed)
    if mu is None:
        mu = np.ones(d)/d
    Sigma = np.identity(d)*sigma
 
    Xt = Xall = np.empty((0, d))
    while np.shape(Xt)[0] < n:

        Xin  = np.random.multivariate_normal(mu, Sigma, 100000)
        Xall = np.vstack((Xall, Xin))
        Xint = Xin[np.linalg.norm(Xin, q, 1) < r, :]
        Xt = np.vstack((Xt, Xint))

    Xt = Xt[:n, :]

    if d == 1:
        Px = np.array([[-1.], [1.]])
    else:
        Px = np.random.multivariate_normal(np.zeros(d), np.identity(d), m)
        Px = r*(Px/(np.linalg.norm(Px, q, 1)[:, None]))
        
    return Xt, Px



if __name__ == "__main__":

    # Initialise variables and parameters of example
    n = 900   # sample size of MVN 
    d = 2     # dimensionality of MVN
    m = 200   # no. of boundary points
    q = 2     # order of l_q ball
    sigma = 1 # value of diagonal covariance of MVN
    seed  = None # random seed of experiment

    # if True, then returns a plotly plot (in browser) of the g function for each dimension, only implemented for d<=2
    # note that when True, timings for TKSD are not as accurate (include calculating g(x) over a grid)
    plot_g = False

    if q == 1:
        r = d
    if q == 2:
        r = d**0.53
    
    # Simulate data
    mu = np.ones(d)*0.5
    X, Px = simulate(n=n, d=d, m=m, r=r, q=q, seed=seed, mu=mu, sigma=sigma)
    
    # Set up parametric form of score function
    logp  = lambda x, theta: -((x - theta[:d].flatten())**2).sum(1)/(2)
    dlogp = grad(logp, 0)

    # Run all three methods (TruncSM, bd-KSD, TKSD) and time
    init = np.random.randn(d)
    
    t = time.perf_counter()
    if q == 2:
        theta_tsm = truncsm_l2(X, dlogp, r=r, theta_init = init)
    elif q == 1:
        theta_tsm = truncsm_l1(X, dlogp, r=r, theta_init = init)
    print(f"TruncSM: {time.perf_counter()-t} seconds")
    
    t = time.perf_counter()
    if q == 2:
        theta_bd = bdKSD_l2(X, dlogp, r=r, theta_init = init)
    elif q == 1:
        theta_bd = bdKSD_l1(X, dlogp, r=r, theta_init = init)
    print(f"bdKSD: {time.perf_counter()-t} seconds")

    t = time.perf_counter()
    if plot_g:
        theta_tksd, g = tksd(X, Px, dlogp, theta_init = init.T, plot_g=True)
    else:
        theta_tksd = tksd(X, Px, dlogp, theta_init = init.T, plot_g=False)
    print(f"TKSD: {time.perf_counter()-t} seconds")

    # Print parameter L2 error
    print("TruncSM error:", np.linalg.norm(theta_tsm - mu).round(3))
    print("bdKSD error:", np.linalg.norm(theta_bd - mu).round(3))
    print("TKSD error:", np.linalg.norm(theta_tksd - mu).round(3))
    