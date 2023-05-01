import autograd.numpy as np
import matplotlib.pyplot as plt

import time

from autograd import elementwise_grad as grad
from tqdm.auto import tqdm

from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from shapely.ops import nearest_points

import sys
import os
sys.path.append(os.getcwd())

from library.truncsm import truncsm_fixedg
from library.tksd import tksd

plt.style.use('seaborn-whitegrid')
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Serif",
    "font.size": 18
})

# Organise a polygon (given by a set of points) and re-organise the order 
# to be clockwise around a given centroid
def organise_poly(corners, centroid):
    
    ang = np.zeros(corners.shape[0])
    for Pi in range(corners.shape[0]):
        Pxi = corners[Pi,:]

        adj = np.abs(Pxi[0] - centroid[0])
        hyp = np.linalg.norm(Pxi - centroid, 2)
        ang[Pi] = np.arccos(adj/hyp)
        
        pos = Pxi - centroid;
        if pos[0] > 0 and pos[1] > 0:
            ang[Pi] = np.pi - ang[Pi]
        elif pos[0] > 0 and pos[1] < 0:
            ang[Pi] = np.pi + ang[Pi]
        elif pos[0] < 0 and pos[1] < 0:
            ang[Pi] = 2*np.pi - ang[Pi]
        
    ind = np.argsort(ang)
    return corners[ind, :]
    

# Simulate a dataset for mixed MVN
def simulate(n, d, m, mu, seed, sigma = 1):
    
    np.random.seed(seed)
    
    corners = np.array([
        [-3, -3], [-3, 3], [3, 3], [3, -3]
    ])

    poly = Polygon(organise_poly(corners, mu.mean(0)))

    Xt = np.empty((0, d))
    while np.shape(Xt)[0] < n:

        X = np.empty((0, d))
        for i in range(mu.shape[0]):
            X1 = np.random.multivariate_normal(mu[i, :], np.eye(d)*sigma, n)
            X  = np.vstack((X, X1))

        Xin = np.zeros((0, 2))
        for i in range(len(X)):
            if poly.contains(Point(X[i, :])):
                Xin = np.vstack((Xin, X[i, :]))
                
        Xt = np.vstack((Xt, Xin))

    Xt = Xt[np.random.permutation(len(Xt))[:n], :]
    Px = np.transpose([poly.exterior.interpolate(t).xy for t in np.linspace(0, poly.length, m, False)])[0].T

    return Xt, Px, poly

# Given a set of thetas outputted by the model, there is no guarantee the first
# theta corresponds to theta_1, etc., so loop through each value of theta and reorganise
# them so the closest theta_hat to a given theta_star is ordered in the same way
def change_to_closest_thetas(thetahat, thetastar):
    num_theta, d = thetastar.shape

    thetahat_compare = thetahat.copy()
    new_order = np.empty(0, dtype=int)
    for i in range(num_theta):
        thetastar_i = thetastar[i, :]
        dist = np.linalg.norm(thetahat_compare - thetastar_i, 2, 1)
        new_order = np.append(new_order, np.argmin(dist))
        thetahat_compare[np.argmin(dist), :] = np.inf

    return thetahat[new_order, :]

# Run the experiment one time: simulate data and estimate parameters with TKSD and MLE
def run_once(n, d, m, num_mixtures=2, seed=None):

    # Set seed
    np.random.seed(seed)

    # True parameters (mu = 1.5)
    sigma = 1
    mus = np.array([[a, b] for a in [-1, 1] for b in [-1, 1]])
    mus = mus[[0, 3, 1, 2]]
    mu  = mus[:num_mixtures, :]*1.5 

    X, dV, poly = simulate(n, d, m, mu, seed)    
    
    start_d = d*(np.arange(num_mixtures)+1)-d
    end_d   = d*(np.arange(num_mixtures)+1)

    # Set up parametric form of score function
    def inlogp(X, theta):
        return np.exp(-np.sum(
            ((X - theta)**2)/(2*sigma**2), 1
        ))
    
    def logp(X, theta):
        return np.log(np.array([
            inlogp(X, theta[start_d[i]:end_d[i]]) for i in range(num_mixtures)
        ]).sum(0))

    dlogp = grad(logp, 0)

    # Get projections for TruncSM using polygon functionality from shapely
    ProjX = np.empty((n, d))
    for i in range(n):
        ProjX[i, :], _ = nearest_points(poly.exterior, Point(X[i,:]))
    
    # get g(x) and its derivative for TruncSM
    g  = np.linalg.norm(X - ProjX, 2, 1)[:, None]
    dg = (X - ProjX)/g

    # Initialise at a noisy version of the true parameter
    init = mu.flatten() + 0.5*np.random.randn(len(mu.flatten()))
    
    # Estimate for TKSD
    t = time.perf_counter()
    theta_tksd = tksd(X, dV, dlogp, theta_init = init.T)
    t_tksd = time.perf_counter() - t

    # Esitmate for TruncSM
    t = time.perf_counter()
    theta_tsm  = truncsm_fixedg(X, g, dg, dlogp, theta_init=init)
    t_tsm = time.perf_counter() - t

    # Reorganise each estimate of mu to its closest counterpart in theta_star
    theta_tksd = change_to_closest_thetas(theta_tksd.reshape(num_mixtures, d), mu)
    theta_tsm  = change_to_closest_thetas(theta_tsm.reshape(num_mixtures, d), mu)

    # Return errors and time taken to run each method
    return (
        np.linalg.norm(theta_tksd.flatten() - mu.flatten()),
        np.linalg.norm(theta_tsm.flatten() - mu.flatten()),
        t_tksd, t_tsm
    )


if __name__ == "__main__":

    # Initialise experiment setup
    n = 200
    d = 2
    m = 200
    seed = 1

    # == Experiment 1: Plot setup of future experiments

    mu = np.array([[a, b] for a in [-1, 1] for b in [-1, 1]])*2
    mu = mu[[0, 3, 1, 2]]

    # Loop over 
    fig, ax = plt.subplots(1, 3, figsize=(9, 2.5))
    for axi, i in enumerate(range(2, 5)):

        mu_i = mu[:i, :]
        X, dV, poly = simulate(n, d, m, mu_i, seed)
        Xall = np.empty((0, d))

        for j in range(mu_i.shape[0]):
            Xj   = np.random.multivariate_normal(mu_i[j, :], np.eye(d), n)
            Xall = np.vstack((Xall, Xj))
        
        Xout = np.zeros((0, 2))
        for j in range(len(Xall)):
            if not poly.contains(Point(Xall[j, :])):
                Xout = np.vstack((Xout, Xall[j, :]))
                
        ax[axi].scatter(Xout[:, 0], Xout[:, 1], c="grey", s=4.5, label="{Untruncated $X$}")
        ax[axi].scatter(X[:, 0], X[:, 1], c="k", s=20, label="{Truncated $X$}")
        ax[axi].plot(*poly.exterior.xy, c="blue", lw=3, label="{$\partial V$}")
        ax[axi].set_title(f"{i} modes")

    ax[axi].legend(loc = "upper left", bbox_to_anchor=(1.03, 1.1),
    frameon=True, handletextpad=0.1, fontsize=18);


    # == Experiment 2: Run benchmark for n

    # Set up variables
    nseq    = np.arange(300, 2100, 300)
    ntrials = 32
    output  = np.empty((ntrials, len(nseq), 4))

    # For each value of n, independently run ntrials experiments to average
    for ni, n in tqdm(enumerate(nseq), total=len(nseq)):
        for seed in range(ntrials):
            output[seed, ni, :] = run_once(int(n), d, 50, 2, seed)

    # Get errors and times from output
    errors = output[:, :, :2]
    times  = output[:, :, 2:]

    # Take mean and standard error to plot
    emeans = errors.mean(0)
    esds   = errors.std(0)/np.sqrt(ntrials)
    tmeans = times.mean(0)
    tsds   = times.std(0)/np.sqrt(ntrials)

    # Plot
    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    ax[0].errorbar(nseq, emeans[:, 0], esds[:, 0], fmt = "-o", c="b", label="\\textbf{TKSD}", lw=1)
    ax[0].errorbar(nseq+15, emeans[:, 1], esds[:, 1], fmt = "-^", c="r", label="\\textbf{TruncSM (exact)}", lw=1)
    ax[0].set_xticks(nseq)
    ax[0].set_xlabel("$n$")
    ax[0].set_ylabel("$\|{\hat{\mu}} - {\mu}^*\|_2$")

    ax[1].errorbar(nseq, tmeans[:, 0], tsds[:, 0], fmt = "-o", c="b", label="\\textbf{TKSD}", lw=1)
    ax[1].errorbar(nseq+15, tmeans[:, 1], tsds[:, 1], fmt = "-^", c="r", label="\\textbf{TruncSM (exact)}", lw=1)
    
    ax[1].set_xticks(nseq)
    ax[1].set_xlabel("$n$")
    ax[1].set_ylabel("Runtime (seconds)")
    ax[1].set_ylim(0, ax[1].get_ylim()[1]+0.22)
    ax[1].legend(loc="upper left");

    plt.subplots_adjust(wspace=.3, hspace=0)
    plt.show()


    # == Experiment 3: Run benchmark for 2 - 4 mixtures
    
    # Set up variables
    ntrials    = 64
    mixtureseq = np.array([2, 3, 4])
    errors     = np.empty((ntrials, 3, 2))

    # For each number of mixture modes, repeat experiment ntrials times
    for i, num_mixture in enumerate(mixtureseq):
        for seed in tqdm(range(ntrials)):
            errors[seed, i, :] = run_once(n, 2, m, num_mixture, seed)
    
    # Take mean and standard error to plot
    means = errors.mean(0)
    sds   = errors.std(0)/np.sqrt(ntrials)

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.errorbar(mixtureseq,      means[:, 0], sds[:, 0], fmt = "-o", c="b", label="\\textbf{TKSD}", lw=1)
    ax.errorbar(mixtureseq+0.1,  means[:, 1], sds[:, 1], fmt = "-^", c="r", label="\\textbf{TruncSM (exact)}", lw=1)

    ax.set_xticks(mixtureseq)
    ax.set_xlabel("Number of Mixtures")
    ax.set_ylabel("$\|{\hat{\mu}} - {\mu}^*\|_2$")
    ax.legend(loc="upper left")
    ax.set_ylim(errors.min()-0.4, errors.max()+0.8)

    plt.subplots_adjust(wspace=.3, hspace=0)
    fig.tight_layout()
    plt.show()