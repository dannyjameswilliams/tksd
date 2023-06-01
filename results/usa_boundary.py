import matplotlib.pyplot as plt
import autograd.numpy as np
import pandas as pd

from shapely.geometry.polygon import Polygon
from shapely.geometry import Point

plt.style.use('seaborn-whitegrid')
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Helvetica",
    "font.size": 18
})

# Re-order the boundary points clockwise from an estimated centroid
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


# Read data
bounds = pd.read_csv("data/america_bounds.csv").values[:, 1:]

# Organise clockwise and convert to Polygon class (for truncation)
usa_poly = Polygon(organise_poly(bounds))

# Simulate full dataset
mu = np.array([-115, 35])
sigma = 10
Sigma = np.eye(2)*sigma
X  = np.random.multivariate_normal(mu, Sigma, 1000)

# Truncate dataset to Polygon
Xt = np.zeros((0, 2))
for i in (range(len(X))):
    if usa_poly.contains(Point(X[i,:])):
        Xt = np.vstack((Xt, X[i, :]))


## USA increasing boundary size plots
mseq = [25, 100, 200]
fig, ax = plt.subplots(1, 3, figsize=(9, 2))
for i in range(3):
    
    bounds_sub = sample_circle(bounds, mseq[i])

    ax[i].scatter(Xt[:,0], Xt[:,1], s=0.3, c="k")
    ax[i].scatter(bounds_sub[:,0], bounds_sub[:,1], s=6, c="#0e89cf")
    ax[i].set_title(f"$m={mseq[i]}$", fontsize=22)
    ax[i].set_xlim(-126, -68)
    ax[i].set_ylim(23, 51)
    if i > 0:
        ax[i].set_yticklabels([])

# Plot legend separately
ax[i].scatter([], [], c="k", s=10, label="$X$")
ax[i].scatter([], [], c="#0e89cf", s=20, label="$\widetilde{\partial V}$")
ax[i].legend(
    loc='upper right', bbox_to_anchor=(1.73, 1.1),
    frameon=True, handletextpad=0.1, fontsize=22
)

plt.subplots_adjust(wspace=.05, hspace=0)
plt.show()