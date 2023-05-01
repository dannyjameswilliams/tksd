import autograd.numpy as np
import matplotlib.pyplot as plt

import seaborn as sns
import pandas as pd

from autograd import elementwise_grad as grad
from tqdm.auto import tqdm

plt.style.use('seaborn-whitegrid')
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Serif",
    "font.size": 18
})

import sys
import os
sys.path.append(os.getcwd())

from library.truncsm import truncsm_l2
from library.bdksd import  bdKSD_l2
from library.tksd_v_statistic import tksd


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


def run(seed, n, d, m, just_tksd=False):

    q = 2     # order of l_q ball
    sigma = 1 # value of diagonal covariance of MVN
    
    r = d**0.53
    
    # Simulate data
    mu = np.ones(d)*0.5
    X, Px = simulate(n=n, d=d, m=m, r=r, q=q, seed=seed, mu=mu, sigma=sigma)
    
    # Set up parametric form of score function
    logp  = lambda x, theta: -((x - theta[:d].flatten())**2).sum(1)/(2)
    dlogp = grad(logp, 0)

    # Run all three methods (TruncSM, bd-KSD, TKSD) and time
    init = np.random.randn(d)
    
    if just_tksd:
        theta_tksd = tksd(X, Px, dlogp, theta_init = init.T)
        return np.linalg.norm(theta_tksd - mu, 2)
    else:
        theta_tsm = truncsm_l2(X, dlogp, r=r, theta_init = init)
        theta_bd = bdKSD_l2(X, dlogp, r=r, theta_init = init)
        theta_tksd = tksd(X, Px, dlogp, theta_init = init.T)
        return np.linalg.norm(theta_tsm - mu, 2), np.linalg.norm(theta_bd - mu, 2), np.linalg.norm(theta_tksd - mu, 2)

def consistency_n(ntrials):

    # Set up variables and constants
    nlist = np.arange(50, 1750, 250)
    d = 2         # fixed dimension

    m = 8*d**2  # how does m scale with dimension? We choose 8d^2, but this is arbitrary
    
    # Loop over n
    errors = np.zeros((ntrials, len(nlist), 3))
    for ni, n in enumerate(nlist):
        
        print(f"n={n} started ({ni+1}/{len(nlist)})")
        
        # Loop over different seeds
        for seed in tqdm(range(ntrials)):
            errors[seed, ni, :] = run(seed, n, d, m)
        
    # Take mean and standard error across errors and runtimes
    means = errors.mean(0)
    sds   = errors.std(0)/np.sqrt(ntrials)
    
    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.errorbar(nlist+50,  means[:, 0], sds[:, 0], fmt = "-^", c="r", label="TruncSM (exact)", lw=1)
    ax.errorbar(nlist+100, means[:, 1], sds[:, 1], fmt = "-s", c="g", label="bd-KSD (exact)", lw=1)
    ax.errorbar(nlist,     means[:, 2], sds[:, 2], fmt = "-o", c="b", label="TKSD", lw=1)

    ax.set_xticks(nlist)
    ax.set_xlabel("$n$")
    ax.set_ylabel("$\|{\hat{\mu}} - {\mu}^*\|_2$")
    ax.legend()

    plt.subplots_adjust(wspace=.3, hspace=0)
    plt.show()


def consistency_n_and_m(ntrials):
    
    # Set up variables and constants
    nlist = np.arange(100, 2000, 250)
    mlist = np.arange(5, 50, 5)
    d = 2        # fixed dimension
    
    # Loop over n
    errors = np.zeros((ntrials, len(nlist), len(mlist), 3))
    for ni, n in enumerate(nlist):
        
        print(f"n={n} started ({ni+1}/{len(nlist)})")
        
        for mi, m in enumerate(mlist):
    
            print(f"m={m} started ({mi+1}/{len(mlist)})")

            # Loop over different seeds
            for seed in tqdm(range(ntrials)):
                errors[seed, ni, mi, :] = run(seed, n, d, m, just_tksd=True)
    
    # Take mean and standard error across errors and runtimes
    means = errors.mean(0)

    # Convert to pandas dataframe for plotting heatmap
    Z_means = pd.DataFrame(means[:, :, 2], index=nlist, columns=mlist)
    
    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(7, 4))

    sns.heatmap(Z_means, ax=ax, cmap="viridis")
    ax.set_ylabel("$n$")
    ax.set_xlabel("$m$")
    
    plt.show()

if __name__ == "__main__":
    
    consistency_n(64)
    consistency_n_and_m(64)
    