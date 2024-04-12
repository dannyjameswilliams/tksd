import pandas as pd
import autograd.numpy as np
import matplotlib.pyplot as plt
import kgof.util as util

from autograd import elementwise_grad as grad

from scipy.linalg import solve_triangular
from scipy.optimize import minimize
from scipy.special import erf
from scipy.stats import norm

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from kgof.kernel import KGauss
from tqdm.auto import tqdm
from autograd import elementwise_grad as grad

plt.style.use('seaborn-whitegrid')
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Serif",
    "font.size": 18
})
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

# For simplicity in this experiment, keep self contained and rewrite
# TKSD library functions (e.g. objective function)

# Make kernel matrices
def kgof_make_Ks(X, Px, bandwidth = 1):
    
    n, d = X.shape
    ndV = Px.shape[0]

    kG = KGauss(bandwidth)
    K = kG.eval(X, X)

    K_y = []
    for i in range(d):
        K_y.append(kG.gradY_X(X, X, i))

    
    Kyt = kG.eval(X, Px)
    Kxtyt = kG.eval(Px, Px)

    Kyt_x = []
    for i in range(d):
        Kyt_x.append(kG.gradX_Y(X, Px, i))

    return K, K_y, Kyt, Kyt_x, Kxtyt

# Objective function for TKSD
def obj_vec(y, X, theta, dlogp, K, K_y, Kyt, Kyt_x, Kxtyt_inv):
    n = len(X)
    dlogpx = dlogp(y, X, theta)

    ksd1 = ((dlogpx @ dlogpx.T) * K).sum()
    ksd2 = 2*(dlogpx.T @ K_y[0]).sum()

    tksd1 = (dlogpx.T @ Kyt)
    tksd3_comb = ((tksd1.T @ tksd1 + 2*(Kyt_x @ tksd1)) * Kxtyt_inv).sum()

    return ((ksd1+ksd2)/(n**2) - tksd3_comb/(n**2))

# Cholesky matrix inverse
def cholesky_inv(A, reg):
    L  = np.linalg.cholesky(A + reg*np.eye(len(A)))
    Linv = solve_triangular(L.T, np.eye(len(L)))
    return Linv @ Linv.T

# Fitting function for conditional density function for TKSD
def tksd(y, X, Px, dlogp, bandwidth=None, theta_init=None):

    d = (X).shape[1]

    if bandwidth is None:
        bandwidth = util.meddistance(y, subsample=1000)**2

    K, K_y, Kyt, Kyt_x, Kxtyt = kgof_make_Ks(y, Px, bandwidth)
    Kxtyt_inv = cholesky_inv(Kxtyt, 1e-6)
    Kyt_x0 = np.stack(Kyt_x, 2).sum(0)

    if theta_init is None:
        theta_init = np.random.randn(d)+1

    obj0 = lambda par: obj_vec(y, X, par, dlogp, K, K_y, Kyt, Kyt_x0, Kxtyt_inv)
    theta_grad = grad(obj0)
    opt = dict(maxiter=1e5, disp=False)
    par = minimize(obj0, theta_init, options=opt, method="BFGS", jac = theta_grad)

    return  par.x

# Auxilliary functions for getting likelihood of Truncated Normal distribution
def phi_trunc(xi):
    return (1 / np.sqrt(2*np.pi)) * np.exp(-0.5*xi**2)

def Phi_trunc(xi):
    return 0.5 * (1 + erf(xi/np.sqrt(2)))

# PDF for truncated Normal distribution
def truncnormpdf(x, mu, sigma, a=-np.inf, b=np.inf):
    alpha = (a-mu)/sigma
    beta  = (b-mu)/sigma
    xi    = (x-mu)/sigma
    return (1/sigma) * (phi_trunc(xi) / (Phi_trunc(beta) - Phi_trunc(alpha)))

# nabla_x log p for regression model
def logp(y, X, theta):
    beta0 = theta[0]
    beta1 = theta[1:]
    mu = beta0 + X @ beta1[:, None]
    return -((y - mu)**2).sum(1)/(2)

def fit_truncnorm(y, X, init, a = -np.inf, b = np.inf, sigma=1):
    
    n = len(y)

    def linear_predictor(beta, i):
        return beta[0] + X[i, :] @ beta[1:][:, None]

    def truncobj(beta):
        out = 0
        for i in range(n):
            out += np.log(truncnormpdf(y[i], linear_predictor(beta, i), sigma, a, b))
        return -out
    
    return minimize(truncobj, init, method="CG").x

def est(y, X, trunc_val, seed = None):

    np.random.seed(seed)
    
    dV = np.array([[trunc_val]])
    init = np.random.randn(X.shape[1] + 1)  

    dlogp = grad(logp, 0)

    beta_tksd = tksd(y, X, dV, dlogp, theta_init=init)

    mod = LinearRegression(fit_intercept=True)
    mod.fit(X, y)
    beta_ls = np.hstack((mod.intercept_.flatten(), mod.coef_.flatten()))

    return beta_tksd, beta_ls

def est_and_plot(y, X, trunc_val, seed=None, true_y=None, true_X=None):

    beta_tksd, beta_ls = est(y, X, trunc_val, seed)

    x = X[:, 0].flatten()

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    if true_y is not None and true_X is not None:
        ax.scatter(true_X[:, 0].flatten(), true_y.flatten(), c="grey", s=10)
        ax.scatter(x, y.flatten(), c="k", s=20)
        x = true_X[:, 0].flatten()
        ax.plot(x, [beta_tksd[0] + beta_tksd[1]*x0 for x0 in x], c="blue", label="TKSD")
        ax.plot(x, [beta_ls[0] + beta_ls[1]*x0 for x0 in x], c="red", label="MLE")
    else:
        ax.scatter(x, y.flatten(), c="k", s=20)
        ax.plot(x, [beta_tksd[0] + beta_tksd[1]*x0 for x0 in x], c="blue", label="TKSD")
        ax.plot(x, [beta_ls[0] + beta_ls[1]*x0 for x0 in x], c="red", label="MLE")
    
    leg = ax.legend(frameon=True, loc="lower right")
    leg.get_frame().set_facecolor('white')

    plt.show()    

    return beta_tksd, beta_ls

def do_simulated(n, seed):

    # Set seed
    np.random.seed(seed)

    # Simulate data
    X = np.random.uniform(0, 1, (n, 1))
    beta0 = 3
    beta1 = 4
    trunc_point = 5 # truncated below at y=5
    y = beta0 + beta1*X + np.random.randn(n, 1)
    trunc = y.flatten() >= trunc_point # take those points above trunc_point
    Xt = X[trunc, :]
    yt = y[trunc, :]

    # Obtain TKSD and least squared (MLE) estimate
    beta_tksd, beta_ls = est(yt, Xt, trunc_point)

    # Predictions on un-truncated points via linear predictor
    tksd_test_pred = beta_tksd[0] + X[~trunc, :] @ beta_tksd[1:, None]
    truncreg_test_pred  = beta_ls[0] + X[~trunc, :] @ beta_ls[1:, None]

    # Log likelihoods on Gaussian (observed)
    tksd_ll_obs = norm.logpdf(y[trunc, :], loc=beta_tksd[0] + X[trunc, :] @ beta_tksd[1:, None]).sum()
    ls_ll_obs = norm.logpdf(y[trunc, :], loc=beta_ls[0] + X[trunc, :] @ beta_ls[1:, None]).sum()

    # Log likelihoods on Gaussian (observed)
    tksd_ll_unobs = norm.logpdf(y[~trunc, :], loc=beta_tksd[0] + X[~trunc, :] @ beta_tksd[1:, None]).sum()
    ls_ll_unobs = norm.logpdf(y[~trunc, :], loc=beta_ls[0] + X[~trunc, :] @ beta_ls[1:, None]).sum()

    # Log likelihoods on truncated Gaussian
    # lp = beta_tksd[0] + Xt @ beta_tksd[1:, None]
    # tksd_ll_trunc = np.log(np.array([truncnormpdf(yt[i], lp[i], 1, trunc_point, np.inf) for i in range(len(Xt))])).sum()
    # lp = beta_ls[0] + Xt @ beta_ls[1:, None]
    # ls_ll_trunc   = np.log(np.array([truncnormpdf(yt[i], lp[i], 1, trunc_point, np.inf) for i in range(len(Xt))])).sum()

    # Return errors and likelihoods
    return (
        mean_squared_error(y[~trunc, :].flatten(), tksd_test_pred.flatten()),
        mean_squared_error(y[~trunc, :].flatten(), truncreg_test_pred.flatten()),
        tksd_ll_obs, ls_ll_obs,
        tksd_ll_unobs, ls_ll_unobs
    )

if __name__ == "__main__":

    # == Experiment 1: Log-likelihood over multiple seeds

    # Repeat the regression experiment over a number of seeds
    ntrials = 256
    output = np.empty((ntrials, 6))
    for i in tqdm(range(ntrials)):
        output[i, :] = do_simulated(600, i)   

    # Get errors and likelihood from experiments
    test_errors = output[:, :2]
    lls = output[:, 2:4]
    llst = output[:, 4:]

    # Plot histograms of results 
    fig, ax = plt.subplots(3, 1, figsize=(6, 7))
    ax[0].hist(test_errors[:, 0], label="{TKSD}", color="b", bins=20, alpha=0.75)
    ax[0].hist(test_errors[:, 1], label="{MLE}", color="r", bins=15, alpha=0.75)
    ax[0].set_title("Error on unobserved points")

    ax[1].hist(lls[:, 0], label="{TKSD}", color="b", bins=20, alpha=0.75)
    ax[1].hist(lls[:, 1], label="{MLE}", color="r", bins=15, alpha=0.75)
    ax[1].set_title("Log-likelihood $\mathcal{N}(\mu, 1)$ (observed)")

    ax[2].hist(llst[:, 0], label="{TKSD}", color="b", bins=20, alpha=0.75)
    ax[2].hist(llst[:, 1], label="{MLE}", color="r", bins=15, alpha=0.75)
    ax[2].set_title("Log-likelihood $\mathcal{N}(\mu, 1)$ (unobserved)")

    ax[2].set_ylim(0, ax[2].get_ylim()[1]+10)
    ax[2].set_xlim(ax[2].get_xlim()[0]-70, ax[2].get_xlim()[1])
    leg = ax[2].legend(frameon=True, loc="lower left")
    leg.get_frame().set_facecolor('white')
    fig.tight_layout()
    
    plt.show()
    



    # == Experiment 2: Example of Regression (one instance of the above)

    # Simulate data
    np.random.seed(1)
    n = 200
    X = np.random.uniform(0, 1, (n, 1))
    beta0 = 3 # true betas
    beta1 = 4
    trunc_point = 5 # truncated at y = 5
    y = beta0 + beta1*X + np.random.randn(n,1)
    trunc = y.flatten() >= trunc_point
    Xt = X[trunc, :]
    yt = y[trunc, :]

    # Estimate and plot with TKSD and Gaussian distribution
    beta_tksd, beta_ls = est_and_plot(yt, Xt, trunc_point, None, y, X);
    

    # - Print likelihoods

    # Log-likelihood for TKSD and least squares on Gaussian distribution (across all y and X)
    tksd_ll = norm.logpdf(y, loc = beta_tksd[0] + X @ beta_tksd[1:, None]).sum()
    ls_ll   = norm.logpdf(y, loc = beta_ls[0] + X @ beta_ls[1:, None]).sum()

    # Log-likelihood for TKSD and least squares on truncated Gaussian distribution (across truncated y and X)
    lp = beta_tksd[0] + Xt @ beta_tksd[1:, None]
    tksd_ll_trunc = np.log(np.array([truncnormpdf(yt[i], lp[i], 1, trunc_point, np.inf) for i in range(len(Xt))])).sum()
    lp = beta_ls[0] + Xt @ beta_ls[1:, None]
    ls_ll_trunc   = np.log(np.array([truncnormpdf(yt[i], lp[i], 1, trunc_point, np.inf) for i in range(len(Xt))])).sum()

    print(f"TKSD log-likelihood: {tksd_ll}" )
    print(f"Least squares log-likelihood: {ls_ll}" )
    
    print(f"TKSD truncated Normal log-likelihood: {tksd_ll_trunc}" )
    print(f"Least squares truncated Normal log-likelihood: {ls_ll_trunc}" )




    # == Experiment 3: Real Data
    
    # Load dataset, these data are truncated at achiv = 40
    df = pd.read_stata("https://stats.idre.ucla.edu/stat/data/truncreg.dta")

    # Split into training and testing
    n = len(df)
    n_train = int(n*0.8)
    n_test  = int(n*0.2)
    ind = np.random.permutation(n)
    train_ind = ind[:n_train]
    test_ind  = ind[n_train:]

    # Conver to y and X numpy arrays 
    y = df.achiv.values[:, None] # response variable is 'achiv' score
    X = df[["langscore"]].values # use single covariate, untruncated score in a different test

    # Estimate TKSD and MLE beta
    beta_tksd, beta_mle = est(y, X, 40)

    # Plot data and regression fit
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    x = X[:, 0].flatten()
    ax.scatter(x, y.flatten(), c="k", s=20) 
    ax.plot(x, [beta_tksd[0] + beta_tksd[1]*x0 for x0 in x], c="blue", label="TKSD")
    ax.plot(x, [beta_mle[0] + beta_mle[1]*x0 for x0 in x], c="red", label="MLE")
    leg = ax.legend(frameon=True, loc="lower right")
    leg.get_frame().set_facecolor('white')
    plt.show()

    
