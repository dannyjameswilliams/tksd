import autograd.numpy as np
from autograd import elementwise_grad as grad

import pandas as pd # for reading CSV
import matplotlib.pyplot as plt

# for proper polygon/point geometry
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

import sys
import os
sys.path.append(os.getcwd())

from library.truncsm import truncsm_proj

# Either the U-statistic or V-statistic form of TKSD (comment out)
from library.tksd_v_statistic import tksd
# from library.tksd_u_statistic import tksd


# Function to organise the polygon coordinates of the USA so they are correctly ordered
def organise_poly(Px):
    centroid = [-100, 40]

    angles = np.zeros(Px.shape[0])
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
        angles[Pi] = ang
        
    ind = np.argsort(angles)
    return Px[ind, :]

# Sample boundary points from a uniform distribution on the unit circle
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
    sampled_angles = np.linspace(0, 2*np.pi, m)
    sampled_angles = np.random.uniform(0, 2*np.pi, m)
    
    # dists = np.zeros(m)
    out_Px = np.zeros((m, Px.shape[1]))
    for i in range(m):
        dist   = np.abs(px_angles - sampled_angles[i])
        out_Px[i, :] = Px[dist.argmin(), :]
        Px = np.delete(Px, dist.argmin(), 0)
        px_angles = np.delete(px_angles, dist.argmin())
    return out_Px 

if __name__ == "__main__":

    # Initialise variables and parameters of example
    n = 400    # sample size of MVN 
    m = 200    # no. of boundary points
    sigma = 10 # value of diagonal covariance of MVN
    
    mu = np.array([-115, 35])  # mean of Gaussian
    
    np.random.seed(1) # random seed of experiment

    # Read/sample data, organise with polygon ordering
    bounds   = pd.read_csv("data/america_bounds.csv").values[:, 1:]
    usa_poly = Polygon(organise_poly(bounds))

    # Simulate full dataset    
    X = np.random.multivariate_normal(mu, np.eye(2)*sigma, 1000)

    # Rejection sampling with Polygon to truncate
    Xt = np.zeros((0, 2))
    for i in (range(len(X))):
        if usa_poly.contains(Point(X[i,:])):
            Xt = np.vstack((Xt, X[i, :]))

    # Set up parametric score function
    logp  = lambda x, theta: -((x - theta.flatten())**2).sum(1)/(2*sigma)
    dlogp = grad(logp, 0)

    # Estimate with TKSD/TruncSM
    init = Xt.mean(0) # good initial guess of mu (truncated mean)
    theta_tksd = tksd(Xt, bounds[:m, :], dlogp, theta_init=init)
    theta_tsm  = truncsm_proj(Xt, bounds[:m, :], dlogp, theta_init=init)


    # Plot
    plt.style.use('seaborn-whitegrid')
    plt.rcParams["font.size"] = 16
    plt.rcParams["text.usetex"] = True

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    plt.scatter(Xt[:,0], Xt[:,1], s=2, c="k")
    plt.scatter(bounds[:,0], bounds[:,1], s=2, c="#595959")
    plt.scatter(theta_tksd[0], theta_tksd[1], label = "TKSD", c="b", s=100)
    plt.scatter(theta_tsm[0], theta_tsm[1], label = "TruncSM", c="r", s=100)
    plt.scatter(mu[0], mu[1], label = "True", c="y", marker="*", s=100)
    plt.legend()

    plt.show()


    print(f"TKSD error: {np.sqrt(np.sum((theta_tksd - mu)**2))}")
    print(f"TruncSM error: {np.sqrt(np.sum((theta_tsm - mu)**2))}")
    print(f"MLE error: {np.sqrt(np.sum((Xt.mean(0) - mu)**2))}")

