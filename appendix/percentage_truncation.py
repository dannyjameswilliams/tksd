import numpy as np
import matplotlib.pyplot as plt

plt.style.use('seaborn-whitegrid')
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Helvetica",
    "font.size": 18
})

# Simulate from l_q ball, given radius r, and mean mu
# and get the mean number of points that were truncated
def simulate_ball(n=30, d=2, ndV=10, r=1, mu=None, sigma=1, q=2):

    if mu is None:
        mu = np.ones(d)
    Sigma = np.identity(d)*sigma
 
    perc = 0

    Xt = Xall = np.empty((0, d))
    while np.shape(Xt)[0] < n:

        Xin  = np.random.multivariate_normal(mu, Sigma, 100000)
        Xall = np.vstack((Xall, Xin))
        Xint = Xin[np.linalg.norm(Xin, q, 1) < r, :] 
        Xt = np.vstack((Xt, Xint))

        perc += (np.linalg.norm(Xin, q, 1) < r).mean()

    Xt = Xt[:n, :]

    if d == 1:
        Px = np.array([[-r], [r]])
    else:
        Px = np.random.multivariate_normal(np.zeros(d), np.identity(d), ndV)
        Px = r*(Px/(np.linalg.norm(Px, q, 1)[:, None])) 

    return Xt, Px, perc

# Run the above function for a given set of parameters
def get_perc(seed, d, q, r, sigma, mu):
    np.random.seed(seed)
    _, __, perc = simulate_ball(n=200, d=d, ndV=1, r=r, mu=mu, sigma=sigma, q=q)
    return perc


# Main function file, loop over different values of dimension d, and get percentage for each one
# repeat experiment over different seeds to take the average and standard deviation
def run(q, ax):
    
    dseq = np.array([2, 4, 6, 8, 10, 12, 14, 16, 18, 20])

    ntrials = 32
    percs = np.empty((ntrials, len(dseq)))
    for i, d in enumerate(dseq):

        mu = np.ones(d)*0.5
        # if q == 1:
        #     mu = np.sqrt(mu)

        if q == 1:
            r = d**0.98
        if q == 2:
            r = d**0.53

        for j in range((ntrials)):
            if q == 1:
                percs[j, i] = get_perc(j, d, q, r, 1, mu)
            elif q == 2:
                percs[j, i] = get_perc(j, d, q, r, 1, mu)

    percs = percs*100

    ax.scatter(dseq, percs.mean(0), c = "k")
    ax.errorbar(dseq, percs.mean(0), percs.std(0), c = "k", label="Percentage of points inside ball")
    
if __name__ == "__main__":

    fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    
    run(1, ax)
    ax.set_xlabel("$d$")
    ax.set_ylabel("Mean \% of points  in ball vs overall")
    ax.set_ylim(58, 68)


    fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    run(2, ax)
    ax.set_ylabel("Mean \% of points  in ball vs overall")
    ax.set_xlabel("$d$")
    ax.set_ylim(40, 60)

    plt.subplots_adjust(wspace=0.1, hspace=0)
    plt.show()

    
