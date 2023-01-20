import matplotlib.pyplot as plt
import autograd.numpy as np
import pandas as pd

from tqdm.auto import tqdm
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from autograd import elementwise_grad as grad

import sys
import os
sys.path.append(os.getcwd())

from library.tksd_v_statistic import tksd
from library.truncsm import truncsm_proj

plt.style.use('seaborn-whitegrid')
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    'text.latex.preamble' : r'\usepackage{amsmath}',
    "font.size": 14
})

# Re-order the boundary points clockwise from an estimated centroid
def organise_poly(Px):
    centroid = [-100, 40]

    ang = np.zeros(Px.shape[0])
    for Pi in range(Px.shape[0]):
        Pxi = Px[Pi,:]

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
    return Px[ind, :]

# Sample uniform points from circle and compare to the boundary points
def sample_circle(Px, m):
    centroid = [-100, 40]
    
    # Get all angles from centroid for Px
    px_angles = np.zeros(Px.shape[0])
    for Pi in range(Px.shape[0]):
        Pxi = Px[Pi,:]

        opp = np.abs(Pxi[0] - centroid[0])
        adj = np.abs(Pxi[1] - centroid[1])
        ang = np.arctan(opp/adj)

        # check domain (+-, ++, --, -+)
        check = Pxi - centroid
        if check[0] < 0 and check[1] < 0:
            ang = ang + np.pi
        elif check[0] > 0 and check[1] < 0:
            ang = ang + np.pi/2
        elif check[0] < 0 and check[1] > 0:
            ang = ang + 3*np.pi/2

        px_angles[Pi] = ang
        
    # find closest angles to sampled angles
    sampled_angles = np.random.uniform(0, 2*np.pi, m)

    out_Px = np.zeros((m, Px.shape[1]))
    for i in range(m):
        dist   = np.abs(px_angles - sampled_angles[i])
        out_Px[i, :] = Px[dist.argmin(), :]

    return out_Px 

# Simulate full dataset from MVN, then truncate until n points from USA
def sim_usa(n, mu, Sigma):

    bounds = pd.read_csv("data/america_bounds.csv").values[:, 1:]
    usa_poly = Polygon(organise_poly(bounds))

    X  = np.random.multivariate_normal(mu, Sigma, 500)
    Xt = np.zeros((0, 2))
    n0 = 0
    while n0 < n:
        for i in range(len(X)):
            if usa_poly.contains(Point(X[i, :])):
                Xt = np.vstack((Xt, X[i, :]))
        n0 = len(Xt)
    Xt = Xt[:n, :]

    return Xt, bounds
            

def exp(seed, n, m):

    # Fix seed for each experiment
    np.random.seed(seed)
    init = np.random.randn(2)

    # Fixed mean and variance of the MVN
    mu = np.array([-115, 35])
    sigma = 10
    Sigma = np.eye(2)*sigma
    Xt, bounds = sim_usa(n, mu, Sigma)

    # Score function for MVN with different sigma
    logp  = lambda x, theta: -((x - theta.flatten())**2).sum(1)/(2*sigma)
    dlogp = grad(logp, 0)

    # Sample m uniform points from boundary using sampling method described above
    Px_sub = sample_circle(bounds, m)

    # Estimate with TKSD and TruncSM
    theta_tksd = tksd(Xt, Px_sub, dlogp, theta_init=init)
    theta_tsm = truncsm_proj(Xt, Px_sub, dlogp, theta_init=init)

    # Return errors
    return (
        np.sqrt(np.sum((theta_tksd - mu)**2)), 
        np.sqrt(np.sum((theta_tsm - mu)**2))
    )

if __name__ == "__main__":

    # Benchmark over a lot of values of m
    mseq = np.array([15, 35, 50, 100, 150, 200, 250, 300, 350, 400])
    ntrials = 256 # number of repetitions (seeds)
    n = 400       # fixed sample size

    # Loop over number of boundary points
    errors = np.empty((ntrials, len(mseq), 2))
    for i, m in enumerate(mseq):
        
        print(f"m={m} started ({i+1}/{len(mseq)})")

        # Loop over seeds
        for seed in tqdm(range(ntrials)):
            errors[seed, i, :] = exp(seed, n, m)
            
        
    # Take mean and standard error for plotting
    means = errors.mean(0)
    sds   = errors.std(0)/np.sqrt(ntrials)
   
    # Plot 
    fig, ax = plt.subplots(1, 1, figsize=(8, 3.5))
    ax.errorbar(mseq+2, means[:,0], sds[:,0], fmt="-o", lw=1, c = "b", label="TKSD")
    ax.errorbar(mseq, means[:,1], sds[:,1], fmt="-^",  lw=1, c = "r", label="TruncSM")
    ax.set_ylabel("$\| {\hat{\mu}} - \mu^* \|_2$")
    ax.set_xlabel("$m$")
    ax.legend();
    plt.show()