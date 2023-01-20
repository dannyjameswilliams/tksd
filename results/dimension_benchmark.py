import autograd.numpy as np
import matplotlib.pyplot as plt
import time

from autograd import elementwise_grad as grad
from tqdm.auto import tqdm

import sys
import osconda 
sys.path.append(os.getcwd())

from library.truncsm import truncsm_l2, truncsm_proj, truncsm_fixedg, polydist_l1ball
from library.tksd_v_statistic import tksd
from library.bdksd import bdKSD_l2, bdKSD_fixedg, bdKSD_proj

plt.style.use('seaborn-whitegrid')
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 20
})


# Simulate data and boundary points from lq ball
def simulate(n=30, d=2, m=10, r=1, seed=2, q=2, mu = None):

    np.random.seed(seed)
    if mu is None:
        mu = np.ones(d)*0.5
    Sigma = np.identity(d)
 
    Xt = Xall = np.empty((0, d))
    counter = 0
    while np.shape(Xt)[0] < n:

        Xin  = np.random.multivariate_normal(mu, Sigma, 100000)
        Xall = np.vstack((Xall, Xin))
        Xint = Xin[np.linalg.norm(Xin, q, 1) < r, :]
        Xt = np.vstack((Xt, Xint))
        counter += 1
        if counter > 1000:
            print(f"cannot sample: n={Xt.shape[0]}")

    Xt = Xt[:n, :]

    if d == 1:
        Px = np.array([[-1.], [1.]])
    else:
        Px = np.random.multivariate_normal(np.zeros(d), np.identity(d), m)
        Px = r*(Px/(np.linalg.norm(Px, q, 1)[:, None]))
    
    return Xt, Px


# Experiment to get errors from truncSM, given different boundary settings (l2 or l1 ball)
def exp(seed, n, d, m, q, r):
    
    # Simulate data with mu = 0.5
    mu = np.ones(d)*0.5
    X, Px = simulate(n=n, d=d, m=m, r=r, q=q, seed=seed, mu=mu)

    # score function for MVN
    logp  = lambda x, theta: -((x - theta.flatten())**2).sum(1)/2
    dlogp = grad(logp, 0)
    
    # fix seed for each experiment
    np.random.seed(seed)
    init = np.random.randn(d)

    # fit estimation methods for the different cases
    if q == 2:

        t = time.time()
        theta_tsm1 = truncsm_l2(X, dlogp, r=r, theta_init = init)
        t_tsm1 = time.time() - t

        t1 = time.time()
        theta_bdksd1  = bdKSD_l2(X, dlogp, r=r, theta_init = init)
        t_bdksd1 = time.time()-t1
        
        polydist_time = 0

    elif q == 1:

        # for lq ball, exact versions of TruncSM and bdKSD use optimisation to find distance functions
        t = time.time()
        dg, g = polydist_l1ball(X, r)
        polydist_time = time.time() - t

        t1 = time.time()
        theta_tsm1  = truncsm_fixedg(X, g[:, None], dg.T, dlogp, theta_init = init)
        t_tsm1 = time.time()-t1

        t1 = time.time()
        theta_bdksd1  = bdKSD_fixedg(X, g[:, None], dg.T, dlogp, theta_init = init)
        t_bdksd1 = time.time()-t1

    # Projection based methods
    t = time.time()
    theta_tsm_proj = truncsm_proj(X, Px, dlogp, theta_init = init)
    t_tsm_proj = time.time() - t

    t = time.time()
    theta_bdksd_proj = bdKSD_proj(X, Px, dlogp, theta_init = init)
    t_bdksd_proj = time.time() - t

    # TKSD
    t = time.time()
    theta_tksd = tksd(X, Px, dlogp, theta_init = init)
    t_tksd = time.time() - t 

    return (
        np.linalg.norm(theta_tsm1 - mu, 2), 
        np.linalg.norm(theta_tsm_proj - mu, 2), 
        np.linalg.norm(theta_bdksd1 - mu, 2), 
        np.linalg.norm(theta_bdksd_proj - mu, 2), 
        np.linalg.norm(theta_tksd - mu, 2), 
        t_tsm1+polydist_time, t_tsm_proj, t_bdksd1+polydist_time, t_bdksd_proj, t_tksd
    )


if __name__ == "__main__":

    # Set up variables and constants
    dlist = np.array([2, 4, 6, 8, 10, 12])
    ntrials = 256  # number of repetitions (seeds)
    n = 300        # fixed sample size
    
    if len(sys.argv) > 1:
        q = int(sys.argv[1])
        print(f"Starting benchmark for the \ell_{q} ball")
    else:
        q = 2

    m_f = lambda d: 8*d**2  # how does m scale with dimension? We choose 8d^2, but this is arbitrary
    
    # Loop over d
    errors = np.zeros((ntrials, len(dlist), 10))
    for di, d in enumerate(dlist):
        
        print(f"d={d} started ({di+1}/{len(dlist)})")

        d = dlist[di]
        B = d if q==1 else d**0.53
        m = m_f(d)
        
        # Loop over different seeds
        for seed in tqdm(range(ntrials)):
            errors[seed, di, :] = exp(seed, n, d, m, q, B)
        
    # Take mean and standard error across errors and runtimes
    means = errors[:, :, :5].mean(0)
    sds   = errors[:, :, :5].std(0)/np.sqrt(ntrials)
    
    tmeans = errors[:, :, 5:].mean(0)
    tsds   = errors[:, :, 5:].std(0)/np.sqrt(ntrials)

    # Plot
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    ax[0].errorbar(dlist+0.1, means[:, 0], sds[:, 0], fmt = "-^", c="r", label="TruncSM (exact)", lw=1)
    ax[0].errorbar(dlist+0.2, means[:, 1], sds[:, 1], fmt = "-v", c="#f58b00", label="TruncSM (approximate)", lw=1)
    ax[0].errorbar(dlist+0.3, means[:, 2], sds[:, 2], fmt = "-s", c="g", label="bd-KSD (exact)", lw=1)
    ax[0].errorbar(dlist+0.4, means[:, 3], sds[:, 3], fmt = "-d", c="#03fc30", label="bd-KSD (approximate)", lw=1)
    ax[0].errorbar(dlist,     means[:, 4], sds[:, 4], fmt = "-o", c="b", label="TKSD", lw=1)

    ax[0].set_xticks(dlist)
    ax[0].set_xlabel("$d$")
    ax[0].set_ylabel("$\|{\hat{\mu}} - {\mu}^*\|_2$")

    ax[1].errorbar(dlist+0.1, tmeans[:, 0], tsds[:, 0], fmt = "-^", c="r", label="TruncSM (exact)", lw=1)
    ax[1].errorbar(dlist+0.2, tmeans[:, 1], tsds[:, 1], fmt = "-v", c="#f58b00", label="TruncSM (approximate)", lw=1)
    ax[1].errorbar(dlist+0.3, tmeans[:, 2], tsds[:, 2], fmt = "-s", c="g", label="bd-KSD (exact)", lw=1)
    ax[1].errorbar(dlist+0.4, tmeans[:, 3], tsds[:, 3], fmt = "-d", c="#03fc30", label="bd-KSD (approximate)", lw=1)
    ax[1].errorbar(dlist, tmeans[:, 4], tsds[:, 4], fmt = "-o", c="b", label="TKSD", lw=1)

    ax[1].set_xticks(dlist)
    ax[1].set_xlabel("$d$")
    ax[1].set_ylabel("Runtime (seconds)")
    ax[1].set_ylim(0, ax[1].get_ylim()[1]+0.2)
    ax[1].legend(loc="upper left");

    fig.suptitle(f"$\ell_{q}$ ball")

    plt.subplots_adjust(wspace=.3, hspace=0)
    plt.show()
