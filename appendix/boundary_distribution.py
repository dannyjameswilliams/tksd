import matplotlib.pyplot as plt
import autograd.numpy as np

from autograd import elementwise_grad as grad
from tqdm.auto import tqdm

import sys
import os
sys.path.append(os.getcwd())

from library.tksd_v_statistic import tksd

plt.style.use('seaborn-whitegrid')
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Serif",
    "font.size": 16
})


def plot_example(seed, n, m):
    
    np.random.seed(seed)

    # Points weighted towards (1, 0)
    dVx = np.random.randn(m)
    dV_weighted_towards = np.vstack((np.cos(dVx), np.sin(dVx))).T

    # Points weighted away
    dVx = np.random.randn(m) - 1
    dV_weighted_away = np.vstack((np.sin(dVx), np.cos(dVx))).T

    # Boundary uniformly distributed
    dVx = np.random.multivariate_normal(np.zeros(2), np.identity(2), m)
    dV_uniform = (dVx/(np.linalg.norm(dVx, 2, 1)[:, None]))
    
    # Truncated dataset towards (1, 0) (mean = [1,0])
    mu = np.array([1, 0])
    X  = np.random.multivariate_normal(mu, np.eye(2), n)
    Xt = X[np.linalg.norm(X, 2, 1) < 1, :]

    # Plot
    fig, ax = plt.subplots(1, 3, figsize=(10, 3))
    for i in range(3):
        ax[i].scatter(X[:, 0], X[:, 1], s = 3, c="k", label="Unobserved Data", marker="x")
        ax[i].scatter(Xt[:, 0], Xt[:, 1], s = 10, c="k", label="Observed Data")
    
        
    ax[0].scatter(dV_weighted_towards[:, 0], dV_weighted_towards[:, 1], s = 20, c="r")
    ax[0].set_title("Distributed Towards $\mu^*$")

    ax[1].scatter(dV_weighted_away[:, 0], dV_weighted_away[:, 1], s = 30, c="r", label="Distributed Away from the Mean")
    ax[1].set_title("Distributed Away from $\mu^*$")

    ax[2].scatter(dV_uniform[:, 0], dV_uniform[:, 1], s = 30, c="r", label = "Boundary Points")
    ax[2].set_title("Uniform")
    
    for i in range(3):
        ax[i].scatter(mu[0], mu[1], s = 150, c="lime", marker="*", label="$\mu^*$")

    ax[2].legend(
        loc='upper right', bbox_to_anchor=(2.25, 1.1),
        frameon=True, handletextpad=0.1, fontsize=18
    )

    plt.show()


def run(seed, n, m):

    np.random.seed(seed)

    # Points weighted towards (1, 0)
    dVx = np.random.randn(m)
    dV_weighted_towards = np.vstack((np.cos(dVx), np.sin(dVx))).T

    # Points weighted away
    dVx = np.random.randn(m) - 1
    dV_weighted_away = np.vstack((np.sin(dVx), np.cos(dVx))).T

    # Boundary uniformly distributed
    dVx = np.random.multivariate_normal(np.zeros(2), np.identity(2), m)
    dV_uniform = (dVx/(np.linalg.norm(dVx, 2, 1)[:, None]))

    # Truncated dataset towards (1, 0) (mean = [1,0])
    mu = np.array([1, 0])
    X  = np.random.multivariate_normal(mu, np.eye(2), n)
    Xt = X[np.linalg.norm(X, 2, 1) < 1, :]

    init = np.random.randn(2)
    logp  = lambda x, theta: -((x - theta.flatten())**2).sum(1)/(2)
    dlogp = grad(logp, 0)

    theta_weighted_towards = tksd(Xt, dV_weighted_towards, dlogp, theta_init = init)
    theta_weighted_away = tksd(Xt, dV_weighted_away, dlogp, theta_init = init)
    theta_uniform = tksd(Xt, dV_uniform, dlogp, theta_init = init)

    return np.linalg.norm(theta_weighted_towards -mu, 2), np.linalg.norm(theta_weighted_away -mu, 2), np.linalg.norm(theta_uniform- mu, 2)
    

if __name__ == "__main__":
    
    
    ntrials = 256
    n = 400
    m = 30

    plot_example(3, n, m)

    errors = np.empty((ntrials, 3))
    for i in tqdm(range(ntrials)):
        errors[i, :] = run(i, n, m)

    fig, ax = plt.subplots(3, 1, figsize=(5, 5))
    ax[0].hist(errors[:, 0], label="weighted towards", bins=15)
    ax[1].hist(errors[:, 1], label="weighted away", bins=15)
    ax[2].hist(errors[:, 2], label="uniform", bins=15)

    ax[0].set_xlim(errors.min()-0.1, errors.max()+0.1)
    ax[0].set_title("Distributed Towards $\mu^*$")

    ax[1].set_xlim(errors.min()-0.1, errors.max()+0.1)
    ax[1].set_title("Distributed Away from $\mu^*$")

    ax[2].set_xlim(errors.min()-0.1, errors.max()+0.1)
    ax[2].set_title("Uniform")
    fig.tight_layout()
    plt.show()

    print(f"""
    Weighted towards mean: {errors[:, 0].mean().round(4)}
    Weighted away mean: {errors[:, 1].mean().round(4)}
    Uniform mean: {errors[:, 2].mean().round(4)}
    """)
